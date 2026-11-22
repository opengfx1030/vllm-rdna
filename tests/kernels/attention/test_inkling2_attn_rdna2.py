# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""fp64 reference tests for the Inkling-2-Mini gfx1030 attention core.

Ops under test (``csrc/rocm/inkling2_attn_rdna2.cu``, ``torch.ops._rocm_C``):
``inkling2_sconv_static``, ``inkling2_sconv_dynamic``, ``inkling2_conv_state_write``,
``inkling2_kv_prep`` and ``inkling2_attention``.

The oracle is torch/CPU float64 written from the model semantics (the eager
torch path in ``vllm/models/inkling2_loop/amd/attention.py``); the CPU tests
pin the oracle to that path.  The oracle never runs in serving.  The paged
worlds are adversarial on purpose: unreferenced pages, slots past the sequence
end, pages outside the sliding window and the null block are NaN, so a kernel
that reads anything it must not produces NaN.  In the sharp mode "spike" keys
sit on both sides of the window / relative-bias boundary, so an off-by-one in a
mask or in the bias index changes the output by far more than the tolerance.

Run: ``pytest tests/kernels/attention/test_inkling2_attn_rdna2.py`` on a
gfx1030 GPU (the CPU oracle tests run everywhere).
"""

import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

HEAD_DIM = 128
D_REL = 16
WINDOW = 1024
REL_EXTENT = 1024
Q_CLIP = 22.627416997969522
REL_CLIP = 8.0
LOG_ALPHA = 0.1
LOG_N_FLOOR = 8192
RMS_EPS = 1e-6
KV_BLOCK = 64
KSIZE = 4
SPIKE_PERIOD = 64

# Attention tolerance against the fp64 oracle (fp16 roundings applied where
# the torch path applies them).  The fp16 output rounding is 2^-11 relative;
# the tiled kernel's fp16 probabilities (relative 2^-11 per weight, summed
# against V values of magnitude ~1..4) add an absolute floor of ~1e-3 on
# outputs that cancel; the decode kernel keeps fp32 probabilities and stays
# an order of magnitude below.  Bugs this suite guards against (a shifted
# window / bias index, a dropped sink) move the output by 1e-2 .. 1.
ATTN_RTOL = 4e-3
ATTN_ATOL = 8e-4


def _on_gfx1030() -> bool:
    if not torch.cuda.is_available():
        return False
    return "gfx1030" in torch.cuda.get_device_properties(0).gcnArchName


def _ops_registered() -> bool:
    import vllm._custom_ops  # noqa: F401  (loads vllm._rocm_C when it is built)

    return hasattr(torch.ops._rocm_C, "inkling2_attention")


requires_kernels = pytest.mark.skipif(
    not _on_gfx1030() or not _ops_registered(),
    reason="needs a gfx1030 GPU with the inkling2 ops in _rocm_C",
)


def _ops():
    from vllm import _custom_ops as ops

    return ops


def _randn(gen, *shape, scale=1.0):
    return torch.randn(*shape, generator=gen, dtype=torch.float32) * scale


def _rand(gen, *shape):
    return torch.rand(*shape, generator=gen, dtype=torch.float32)


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def to_device_cache(
    cache: torch.Tensor, token_major: bool, layers: int = 1
) -> torch.Tensor:
    """``[B, H, bs, C]`` cache on the GPU.  ``token_major`` stores the pages
    token-major, ``[B, bs, H, C]`` in memory, and returns the permuted
    (non-contiguous) ``[B, H, bs, C]`` view: the engine's ``LBNHC`` / ``BLNHC``
    ``KVCacheLayout`` (``hbc`` is ``LBHNC`` / ``BLHNC``).  ``layers > 1`` stores
    the pages block-outermost between the pages of ``layers - 1`` other layers
    (NaN-filled, block stride = ``layers`` pages), as ``BLHNC`` / ``BLNHC`` do."""
    pages = cache.permute(0, 2, 1, 3) if token_major else cache
    if layers == 1:
        dev = pages.contiguous().cuda() if token_major else cache.cuda()
    else:
        buf = torch.full((pages.shape[0], layers, *pages.shape[1:]), float("nan"))
        buf[:, layers // 2] = pages
        dev = buf.to(cache.dtype).cuda()[:, layers // 2]
    return dev.permute(0, 2, 1, 3) if token_major else dev


def assert_unchanged(tensors, snapshots) -> None:
    """Inputs of an op (the arguments without a ``!`` in its schema) must come
    back bit-identical."""
    for i, (t, s) in enumerate(zip(tensors, snapshots)):
        if t is not None:
            torch.testing.assert_close(
                t, s, rtol=0, atol=0, equal_nan=True, msg=f"arg {i}"
            )


class PagePool:
    """Shuffled page ids; id 0 is the null block (never handed out)."""

    def __init__(self, gen, total: int):
        self.ids = (torch.randperm(total - 1, generator=gen) + 1).tolist()

    def take(self, n: int) -> list[int]:
        out, self.ids = self.ids[:n], self.ids[n:]
        assert len(out) == n
        return out


# ---------------------------------------------------------------------------
# conv oracle
# ---------------------------------------------------------------------------
def conv_static_oracle(taps: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Taps [k, T, C] (taps[lag] = input ``lag`` tokens back), w [C, k]."""
    k = taps.shape[0]
    out = torch.zeros_like(taps[0])
    for lag in range(k):
        out = out + taps[lag] * w[:, k - 1 - lag]
    return (out + taps[0]) / math.sqrt(k + 1)


def conv_dynamic_oracle(
    taps: torch.Tensor, static_weight: torch.Tensor, dyn: torch.Tensor
) -> torch.Tensor:
    """Taps [k, T, C]; static_weight [k, C]; dyn [T, H, k] (per-head taps)."""
    k, T, C = taps.shape
    H = dyn.shape[1]
    out = torch.zeros_like(taps[0])
    for lag in range(k):
        w = static_weight[lag][None, :] + dyn[:, :, lag].repeat_interleave(C // H, 1)
        out = out + taps[lag] * w
    return (out + taps[0]) / math.sqrt(2 * k + 1)


def test_conv_oracle_matches_repo_modules():
    from vllm.models.inkling2_loop.amd.attention import DynamicSconv, StaticSconv

    gen = torch.Generator().manual_seed(0)
    T, C, H = 11, 24, 4
    taps = _randn(gen, KSIZE, T, C)
    static = StaticSconv(C, KSIZE)
    static.conv1d.weight.data.copy_(_randn(gen, C, 1, KSIZE))
    torch.testing.assert_close(
        conv_static_oracle(
            taps.double(), static.conv1d.weight.data.squeeze(1).double()
        ),
        static.forward_taps(taps).double(),
        rtol=1e-5,
        atol=1e-6,
    )
    dynamic = DynamicSconv(C, 16, H, KSIZE)
    dynamic.static_weight.data.copy_(_randn(gen, KSIZE, C))
    dynamic.weight_proj.weight = torch.nn.Parameter(
        _randn(gen, H * KSIZE, 16), requires_grad=False
    )
    h = _randn(gen, T, 16)
    dyn = F.linear(h, dynamic.weight_proj.weight).float().view(T, H, KSIZE)
    torch.testing.assert_close(
        conv_dynamic_oracle(
            taps.double(), dynamic.static_weight.data.double(), dyn.double()
        ),
        dynamic.forward_taps(taps, h).double(),
        rtol=1e-5,
        atol=1e-6,
    )


# ---------------------------------------------------------------------------
# conv worlds
# ---------------------------------------------------------------------------
def conv_keeps(p: int, p1: int, bs: int, ring: int) -> bool:
    """Whether position ``p`` of a chunk that ends at ``p1`` writes its ring row: only
    the last ``ring`` positions of the intersection of the chunk with a block (one
    writer per row; the rows of earlier positions would be overwritten anyway)."""
    return p + ring >= min(p1, (p // bs + 1) * bs)


class ConvWorld:
    """Paged conv-state world for one or more streams of one layer.

    ``streams`` is a list of ``(offset, width, dtype)``; stream s has
    ``C = nkv * width`` channels and ``dtype`` is the dtype of the raw input ``x`` the
    op gets.  The state is fp16 ``[pages, nkv, ring, W]`` (``W`` = the widest
    ``offset + width``): the block ``p // bs`` of a request holds position ``p`` in ring
    row ``p % ring``.  The cache
    holds the (up to) ``k - 1`` history positions of each request -- the whole ring with
    ``fill_ring`` -- everything else is NaN; blocks wholly below the history are the
    null block, as the engine's forward block table shows them.

    Raw inputs are fp16-representable (the serving path's conv inputs are fp16 GEMM
    outputs); ``exact_fp32`` keeps fp32 streams at full precision: the history is then
    read back rounded to fp16, the taps inside the chunk stay exact.
    """

    def __init__(
        self,
        q_lens,
        seq_lens,
        *,
        nkv,
        streams,
        bs,
        ring=KSIZE - 1,
        seed=0,
        pad_tokens=0,
        md_short=0,
        fill_ring=False,
        exact_fp32=False,
    ):
        gen = torch.Generator().manual_seed(seed)
        self.q_lens, self.seq_lens = list(q_lens), list(seq_lens)
        self.nkv, self.streams, self.bs, self.ring = nkv, streams, bs, ring
        self.R = len(q_lens)
        self.n_valid = sum(q_lens)
        self.T = self.n_valid + pad_tokens
        self.T_md = self.T - md_short  # tokens covered by the metadata tensors
        assert self.T_md >= self.n_valid
        self.W = max(o + w for o, w, _ in streams)
        self.qsl = [0]
        for ql in q_lens:
            self.qsl.append(self.qsl[-1] + ql)
        nb = [_ceil_div(L, bs) for L in seq_lens]
        first = [max(0, L - ql - (KSIZE - 1)) // bs for L, ql in zip(seq_lens, q_lens)]
        self.num_pages = sum(max(n - f, 0) for n, f in zip(nb, first)) + 5
        pool = PagePool(gen, self.num_pages)
        self.table = torch.zeros(self.R, max(nb) + 2, dtype=torch.int32)
        for r in range(self.R):
            if nb[r] > first[r]:
                self.table[r, first[r] : nb[r]] = torch.tensor(
                    pool.take(nb[r] - first[r]), dtype=torch.int32
                )
        # dense raw inputs per stream: [L, nkv * width]; every third position is 4x
        # louder, so the k L2 clip (threshold 22.6) engages on those conv outputs
        # (norm ~ 50) and stays off on the others
        loud = [
            1.0 + 3.0 * (torch.arange(L) % 3 == 0).float()[:, None] for L in seq_lens
        ]
        self.x_full = [
            [
                (_randn(gen, L, nkv * width, scale=2.0) * loud[r])
                .to(dt if exact_fp32 else torch.float16)
                .float()
                for r, L in enumerate(seq_lens)
            ]
            for _, width, dt in streams
        ]
        # what the cache holds of a position: the input rounded to fp16
        self.x_hist = [[x.half().float() for x in xs] for xs in self.x_full]
        self.cache0 = torch.full(
            (self.num_pages, nkv, ring, self.W), float("nan"), dtype=torch.float16
        )
        depth = ring if fill_ring else KSIZE - 1
        for r in range(self.R):
            p0 = seq_lens[r] - q_lens[r]
            for p in range(max(0, p0 - depth), p0):
                page = int(self.table[r, p // bs])
                if page == 0:
                    continue  # below the window: the null block
                for s, (off, width, _) in enumerate(streams):
                    self.cache0[page, :, p % ring, off : off + width] = self.x_hist[s][
                        r
                    ][p].view(nkv, width)
        self.seq_idx = torch.full((self.T_md,), -1, dtype=torch.int32)
        for r in range(self.R):
            for i in range(q_lens[r]):
                self.seq_idx[self.qsl[r] + i] = r
        self.qsl_t = torch.tensor(self.qsl, dtype=torch.int32)
        self.seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32)
        # chunk rows of the raw input (NaN rows for padding tokens)
        self.x_rows = []
        for s, (_, width, dt) in enumerate(streams):
            rows = torch.full((self.T, nkv * width), float("nan"))
            for r in range(self.R):
                p0 = seq_lens[r] - q_lens[r]
                rows[self.qsl[r] : self.qsl[r + 1]] = self.x_full[s][r][
                    p0 : p0 + q_lens[r]
                ]
            self.x_rows.append(rows.to(dt))
        # per-stream weights: static [C, 1, k]; dynamic static_weight [k, C]
        # plus the per-token per-head taps [T, nkv * k]
        self.static_w = [
            _randn(gen, nkv * width, 1, KSIZE, scale=0.5) for _, width, _ in streams
        ]
        self.dyn_static_w = [
            _randn(gen, KSIZE, nkv * width, scale=0.5) for _, width, _ in streams
        ]
        self.dyn_w = [
            _randn(gen, self.T, nkv * KSIZE, scale=0.4).half() for _ in streams
        ]

    def taps(self, s: int) -> torch.Tensor:
        """[k, n_valid, C] float64 taps of stream s from the dense inputs: the inputs of
        this chunk exact, the history as the cache returns it (fp16)."""
        C = self.nkv * self.streams[s][1]
        taps = torch.zeros(KSIZE, self.n_valid, C, dtype=torch.float64)
        for r in range(self.R):
            p0 = self.seq_lens[r] - self.q_lens[r]
            for i in range(self.q_lens[r]):
                for lag in range(KSIZE):
                    q = p0 + i - lag
                    if q >= 0:
                        src = self.x_full if q >= p0 else self.x_hist
                        taps[lag, self.qsl[r] + i] = src[s][r][q].double()
        return taps

    def oracle(self, s: int, dynamic: bool) -> torch.Tensor:
        taps = self.taps(s)
        if dynamic:
            dyn = self.dyn_w[s][: self.n_valid].double().view(-1, self.nkv, KSIZE)
            ref = conv_dynamic_oracle(taps, self.dyn_static_w[s].double(), dyn)
        else:
            ref = conv_static_oracle(taps, self.static_w[s].squeeze(1).double())
        out = torch.zeros(self.T, ref.shape[1], dtype=torch.float64)
        out[: self.n_valid] = ref
        return out

    def expected_cache(self, streams=None) -> torch.Tensor:
        """The cache after the chunks were written (``streams``: indices written)."""
        cache = self.cache0.clone()
        for r in range(self.R):
            p0 = self.seq_lens[r] - self.q_lens[r]
            for p in range(p0, self.seq_lens[r]):
                if not conv_keeps(p, self.seq_lens[r], self.bs, self.ring):
                    continue
                page = int(self.table[r, p // self.bs])
                for s, (off, width, _) in enumerate(self.streams):
                    if streams is None or s in streams:
                        cache[page, :, p % self.ring, off : off + width] = self.x_hist[
                            s
                        ][r][p].view(self.nkv, width)
        return cache

    def meta(self):
        dev = "cuda"
        return (
            self.table.to(dev),
            self.seq_idx.to(dev),
            self.qsl_t.to(dev),
            self.seq_lens_t.to(dev),
        )


CONV_CASES = {
    "decode": ([1, 1, 1, 1, 1, 1], [1, 2, 3, 4, 70, 129]),
    "prefill": ([8, 3, 64, 1, 5], [8, 3, 64, 1, 5]),
    "chunked": ([5, 64, 2, 1, 17], [100, 129, 257, 4, 66]),
    "empty": ([2, 0, 1, 7], [9, 0, 33, 7]),
}
PADDINGS = [(0, 0), (5, 3)]  # (padding tokens, tokens missing from the metadata)
# (conv block size, ring rows): the real model has 64 and 3 (10 with 7 draft tokens)
GEOMETRIES = [(4, 3), (64, 3), (16, 10)]


def _assert_cache(cache: torch.Tensor, want: torch.Tensor, what: str) -> None:
    torch.testing.assert_close(
        cache.cpu(), want, rtol=0, atol=0, equal_nan=True, msg=what
    )


@requires_kernels
@pytest.mark.parametrize("case", list(CONV_CASES))
@pytest.mark.parametrize("bs,ring", GEOMETRIES)
@pytest.mark.parametrize(
    "nkv,width,dtype",
    [
        (8, 288, torch.float32),
        (4, 576, torch.float16),
        (2, 128, torch.float16),
        # the post-attention (fp32 o_proj all-reduce) and post-MLP (fp16) streams of
        # a TP=2 / TP=4 rank: 2304 channels over its 4 / 2 kv heads
        (4, 576, torch.float32),
        (2, 1152, torch.float32),
        (2, 1152, torch.float16),
    ],
)
@pytest.mark.parametrize("padding", PADDINGS)
@pytest.mark.parametrize("token_major", [False, True], ids=["hbc", "bhc"])
def test_sconv_static(case, bs, ring, nkv, width, dtype, padding, token_major):
    ops = _ops()
    q_lens, seq_lens = CONV_CASES[case]
    offset = 128
    cw = ConvWorld(
        q_lens,
        seq_lens,
        nkv=nkv,
        streams=[(offset, width, dtype)],
        bs=bs,
        ring=ring,
        fill_ring=ring > KSIZE - 1,
        pad_tokens=padding[0],
        md_short=padding[1],
        seed=3,
    )
    table, seq_idx, qsl, seq_lens_t = cw.meta()
    x, wt = cw.x_rows[0].cuda(), cw.static_w[0].cuda()
    ins = [x, wt, table, seq_idx, qsl, seq_lens_t]
    snap = [t.clone() for t in ins]
    want = cw.oracle(0, False)

    # compute only: the state stays bit-identical; the write is a separate op
    cache = to_device_cache(cw.cache0, token_major)
    out = torch.full((cw.T, nkv * width), float("nan"), device="cuda")
    ops.inkling2_sconv_static(
        x, wt, out, cache, table, seq_idx, qsl, seq_lens_t, offset, nkv, bs, False
    )
    torch.accelerator.synchronize()
    assert_unchanged(ins, snap)
    torch.testing.assert_close(out.cpu().double(), want, rtol=2e-5, atol=2e-5)
    _assert_cache(cache, cw.cache0, "compute-only call touched the state")
    ops.inkling2_conv_state_write(
        x, cache, table, seq_idx, qsl, seq_lens_t, offset, nkv, bs
    )
    torch.accelerator.synchronize()
    assert_unchanged(ins, snap)
    _assert_cache(cache, cw.expected_cache(), "state write")

    if case == "decode":  # one token per request: the op reads, then writes, itself
        cache = to_device_cache(cw.cache0, token_major)
        out = torch.full((cw.T, nkv * width), float("nan"), device="cuda")
        ops.inkling2_sconv_static(
            x, wt, out, cache, table, seq_idx, qsl, seq_lens_t, offset, nkv, bs, True
        )
        torch.accelerator.synchronize()
        assert_unchanged(ins, snap)
        torch.testing.assert_close(out.cpu().double(), want, rtol=2e-5, atol=2e-5)
        _assert_cache(cache, cw.expected_cache(), "fused write")


@requires_kernels
@pytest.mark.parametrize("case", list(CONV_CASES))
@pytest.mark.parametrize("bs,ring", GEOMETRIES)
@pytest.mark.parametrize("nkv", [8, 4, 2])
@pytest.mark.parametrize("padding", PADDINGS)
@pytest.mark.parametrize("token_major", [False, True], ids=["hbc", "bhc"])
def test_sconv_dynamic(case, bs, ring, nkv, padding, token_major):
    ops = _ops()
    q_lens, seq_lens = CONV_CASES[case]
    cw = ConvWorld(
        q_lens,
        seq_lens,
        nkv=nkv,
        streams=[(0, 128, torch.float16)],
        bs=bs,
        ring=ring,
        fill_ring=ring > KSIZE - 1,
        pad_tokens=padding[0],
        md_short=padding[1],
        seed=4,
    )
    table, seq_idx, qsl, seq_lens_t = cw.meta()
    x, sw, dyn = cw.x_rows[0].cuda(), cw.dyn_static_w[0].cuda(), cw.dyn_w[0].cuda()
    ins = [x, sw, dyn, table, seq_idx, qsl, seq_lens_t]
    snap = [t.clone() for t in ins]
    want = cw.oracle(0, True)

    cache = to_device_cache(cw.cache0, token_major)
    out = torch.full((cw.T, nkv * 128), float("nan"), device="cuda")
    ops.inkling2_sconv_dynamic(
        x, sw, dyn, out, cache, table, seq_idx, qsl, seq_lens_t, 0, nkv, bs, False
    )
    torch.accelerator.synchronize()
    assert_unchanged(ins, snap)
    torch.testing.assert_close(out.cpu().double(), want, rtol=2e-5, atol=2e-5)
    _assert_cache(cache, cw.cache0, "compute-only call touched the state")
    ops.inkling2_conv_state_write(x, cache, table, seq_idx, qsl, seq_lens_t, 0, nkv, bs)
    torch.accelerator.synchronize()
    _assert_cache(cache, cw.expected_cache(), "state write")

    if case == "decode":
        cache = to_device_cache(cw.cache0, token_major)
        out = torch.full((cw.T, nkv * 128), float("nan"), device="cuda")
        ops.inkling2_sconv_dynamic(
            x, sw, dyn, out, cache, table, seq_idx, qsl, seq_lens_t, 0, nkv, bs, True
        )
        torch.accelerator.synchronize()
        torch.testing.assert_close(out.cpu().double(), want, rtol=2e-5, atol=2e-5)
        _assert_cache(cache, cw.expected_cache(), "fused write")


# ---------------------------------------------------------------------------
# conv state write
# ---------------------------------------------------------------------------
@requires_kernels
@pytest.mark.parametrize("case", list(CONV_CASES))
@pytest.mark.parametrize("bs,ring", GEOMETRIES)
@pytest.mark.parametrize("padding", PADDINGS)
@pytest.mark.parametrize("token_major", [False, True], ids=["hbc", "bhc"])
@pytest.mark.parametrize("layers", [1, 5], ids=["page", "block5"])
def test_conv_state_write(case, bs, ring, padding, token_major, layers):
    """Two streams of different dtype and width into one page: each write touches only
    its own columns and the rows the rule keeps, whatever the strides of the cache
    (``block5``: the page sits between the pages of four other layers)."""
    ops = _ops()
    q_lens, seq_lens = CONV_CASES[case]
    nkv = 4
    cw = ConvWorld(
        q_lens,
        seq_lens,
        nkv=nkv,
        streams=[(0, 128, torch.float16), (128, 288, torch.float32)],
        bs=bs,
        ring=ring,
        fill_ring=True,
        pad_tokens=padding[0],
        md_short=padding[1],
        seed=7,
    )
    table, seq_idx, qsl, seq_lens_t = cw.meta()
    cache = to_device_cache(cw.cache0, token_major, layers)
    xs = [cw.x_rows[0].cuda(), cw.x_rows[1].cuda()]
    ins = [*xs, table, seq_idx, qsl, seq_lens_t]
    snap = [t.clone() for t in ins]
    ops.inkling2_conv_state_write(
        xs[0], cache, table, seq_idx, qsl, seq_lens_t, 0, nkv, bs
    )
    torch.accelerator.synchronize()
    _assert_cache(cache, cw.expected_cache(streams=[0]), "first stream")
    ops.inkling2_conv_state_write(
        xs[1], cache, table, seq_idx, qsl, seq_lens_t, 128, nkv, bs
    )
    torch.accelerator.synchronize()
    assert_unchanged(ins, snap)
    _assert_cache(cache, cw.expected_cache(), "both streams")


@requires_kernels
@pytest.mark.parametrize("bs,ring", GEOMETRIES)
@pytest.mark.parametrize("nkv,width", [(2, 128), (4, 576), (2, 1152)])
def test_conv_state_rounds_fp32_inputs_to_fp16(bs, ring, nkv, width):
    """An fp32 ``x`` that is not fp16-representable (the post-attention stream of a
    tensor-parallel rank: the fp32 all-reduce of the o_proj partials) is stored
    rounded to nearest even; the taps inside the chunk are exact, the history comes
    back rounded."""
    ops = _ops()
    q_lens, seq_lens = [37, 5, 1], [90, 5, 41]
    cw = ConvWorld(
        q_lens,
        seq_lens,
        nkv=nkv,
        streams=[(0, width, torch.float32)],
        bs=bs,
        ring=ring,
        exact_fp32=True,
        seed=11,
    )
    assert not torch.equal(cw.x_full[0][0], cw.x_hist[0][0])
    table, seq_idx, qsl, seq_lens_t = cw.meta()
    cache = to_device_cache(cw.cache0, False)
    out = torch.full((cw.T, nkv * width), float("nan"), device="cuda")
    x, wt = cw.x_rows[0].cuda(), cw.static_w[0].cuda()
    ops.inkling2_sconv_static(
        x, wt, out, cache, table, seq_idx, qsl, seq_lens_t, 0, nkv, bs, False
    )
    ops.inkling2_conv_state_write(x, cache, table, seq_idx, qsl, seq_lens_t, 0, nkv, bs)
    torch.accelerator.synchronize()
    torch.testing.assert_close(
        out.cpu().double(), cw.oracle(0, False), rtol=2e-5, atol=2e-5
    )
    _assert_cache(cache, cw.expected_cache(), "rounded state")


@requires_kernels
@pytest.mark.parametrize("bs,drafts", [(4, 4), (16, 7), (64, 7)])
@pytest.mark.parametrize("accepted", [1, 2, 4, 8])
@pytest.mark.parametrize("fused_decode", [False, True], ids=["chunk", "decode"])
def test_rejected_draft_tokens_keep_the_history_of_the_accepted_prefix(
    bs, drafts, accepted, fused_decode
):
    """Verify step of ``1 + drafts`` tokens, of which ``accepted`` stay; the junk rows
    of the rejected tokens are in the ring, then the next step reads its history from
    it.  ``ring = k - 1 + drafts`` rows keep every accepted position readable, the
    rejected rows are overwritten by the next step's own writes.  ``fused_decode``:
    the same run with one-token steps (the ops write their own state)."""
    accepted = min(accepted, 1 + drafts)
    ops = _ops()
    ring, nkv, width, p0, after = KSIZE - 1 + drafts, 2, 128, 37, 6
    C = nkv * width
    gen = torch.Generator().manual_seed(bs + accepted)
    n_all = p0 + accepted + after
    truth = (_randn(gen, n_all, C, scale=2.0)).half().float()
    wt = _randn(gen, C, 1, KSIZE, scale=0.5).cuda()
    nblk = _ceil_div(p0 + 1 + drafts, bs) + 2
    pages = torch.randperm(nblk + 4, generator=gen)[:nblk] + 1  # never the null block
    table = torch.zeros(1, nblk + 2, dtype=torch.int32)
    table[0, :nblk] = pages.to(torch.int32)
    cache = torch.full((nblk + 6, nkv, ring, width), float("nan"), dtype=torch.float16)
    for p in range(max(0, p0 - ring), p0):  # the history of the first verify step
        cache[int(table[0, p // bs]), :, p % ring, :] = truth[p].view(nkv, width).half()
    cache_d, table_d = cache.cuda(), table.cuda()

    def step(first, rows, count):
        """Run ``count`` tokens starting at position ``first`` (inputs ``rows``)."""
        x = rows.cuda().contiguous()
        out = torch.full((count, C), float("nan"), device="cuda")
        seq_idx = torch.zeros(count, dtype=torch.int32, device="cuda")
        qsl = torch.tensor([0, count], dtype=torch.int32, device="cuda")
        lens = torch.tensor([first + count], dtype=torch.int32, device="cuda")
        write = fused_decode and count == 1
        ops.inkling2_sconv_static(
            x, wt, out, cache_d, table_d, seq_idx, qsl, lens, 0, nkv, bs, write
        )
        if not write:
            ops.inkling2_conv_state_write(
                x, cache_d, table_d, seq_idx, qsl, lens, 0, nkv, bs
            )
        torch.accelerator.synchronize()
        return out.cpu().double()

    junk = _randn(gen, 1 + drafts - accepted, C, scale=40.0).half().float()
    verify = torch.cat([truth[p0 : p0 + accepted], junk])
    if fused_decode:  # one-token steps: accept the first ``accepted``, then junk ones
        outs = [step(p0 + i, verify[i : i + 1], 1) for i in range(1 + drafts)]
        got_verify = torch.cat(outs[:accepted])
        # rolling back one-token steps: the junk steps wrote ring rows of positions
        # p0 + accepted ..; the next step recomputes position p0 + accepted
    else:
        got_verify = step(p0, verify, 1 + drafts)[:accepted]
    got_next = step(p0 + accepted, truth[p0 + accepted :], after)
    taps_x = truth.double()
    wd = wt.cpu().squeeze(1).double()

    def want(lo, hi):
        taps = torch.zeros(KSIZE, hi - lo, C, dtype=torch.float64)
        for i, p in enumerate(range(lo, hi)):
            for lag in range(KSIZE):
                if p - lag >= 0:
                    taps[lag, i] = taps_x[p - lag]
        return conv_static_oracle(taps, wd)

    torch.testing.assert_close(
        got_verify, want(p0, p0 + accepted), rtol=2e-5, atol=2e-5
    )
    torch.testing.assert_close(
        got_next, want(p0 + accepted, n_all), rtol=2e-5, atol=2e-5
    )


# ---------------------------------------------------------------------------
# kv_prep
# ---------------------------------------------------------------------------
@requires_kernels
@pytest.mark.parametrize("case", list(CONV_CASES))
@pytest.mark.parametrize("bs,ring", GEOMETRIES)
@pytest.mark.parametrize("dynamic", [False, True], ids=["static", "dynamic"])
@pytest.mark.parametrize("nkv", [8, 2])
@pytest.mark.parametrize("padding", PADDINGS)
@pytest.mark.parametrize("token_major", [False, True], ids=["hbc", "bhc"])
def test_kv_prep(case, bs, ring, dynamic, nkv, padding, token_major):
    ops = _ops()
    q_lens, seq_lens = CONV_CASES[case]
    ko, vo = 0, 128
    cw = ConvWorld(
        q_lens,
        seq_lens,
        nkv=nkv,
        streams=[(ko, 128, torch.float16), (vo, 128, torch.float16)],
        bs=bs,
        ring=ring,
        fill_ring=ring > KSIZE - 1,
        pad_tokens=padding[0],
        md_short=padding[1],
        seed=5,
    )
    # K|V cache (page size 64, its own table and slot mapping)
    gen = torch.Generator().manual_seed(6)
    nbk = [_ceil_div(L, KV_BLOCK) for L in seq_lens]
    pool = PagePool(gen, sum(nbk) + 5)
    kv_table = [pool.take(n) for n in nbk]
    kv_cache0 = torch.full(
        (sum(nbk) + 5, nkv, KV_BLOCK, 2 * HEAD_DIM), float("nan"), dtype=torch.float16
    )
    kv_slot = torch.full((cw.T_md,), -1, dtype=torch.int64)
    for r, (ql, L) in enumerate(zip(q_lens, seq_lens)):
        for i in range(ql):
            p = L - ql + i
            kv_slot[cw.qsl[r] + i] = (
                kv_table[r][p // KV_BLOCK] * KV_BLOCK + p % KV_BLOCK
            )
    table, seq_idx, qsl, seq_lens_t = cw.meta()
    if dynamic:
        k_w, v_w = cw.dyn_static_w[0], cw.dyn_static_w[1]
        k_dyn, v_dyn = cw.dyn_w[0].cuda(), cw.dyn_w[1].cuda()
    else:
        k_w, v_w = cw.static_w[0], cw.static_w[1]
        k_dyn = v_dyn = None
    xk, xv, kw, vw = (t.cuda() for t in (cw.x_rows[0], cw.x_rows[1], k_w, v_w))
    kv_slot_d = kv_slot.cuda()
    ins = [xk, xv, kw, vw, k_dyn, v_dyn, table, seq_idx, qsl, seq_lens_t, kv_slot_d]
    snap = [None if t is None else t.clone() for t in ins]

    n = cw.n_valid
    k = cw.oracle(0, dynamic)[:n].view(n, nkv, HEAD_DIM)
    v = cw.oracle(1, dynamic)[:n].view(n, nkv, HEAD_DIM)
    k = k * (Q_CLIP / (torch.linalg.vector_norm(k, dim=-1, keepdim=True) + 1e-6)).clamp(
        max=1.0
    )
    v = v * torch.rsqrt(v.square().mean(-1, keepdim=True) + RMS_EPS)
    rows = torch.cat([k, v], dim=-1).to(torch.float16)
    expected = kv_cache0.clone()
    for r, (ql, L) in enumerate(zip(q_lens, seq_lens)):
        for i in range(ql):
            p = L - ql + i
            expected[kv_table[r][p // KV_BLOCK], :, p % KV_BLOCK, :] = rows[
                cw.qsl[r] + i
            ]

    def run(write_state: bool):
        cache = to_device_cache(cw.cache0, token_major)
        kvc = to_device_cache(kv_cache0, token_major)
        ops.inkling2_kv_prep(
            xk, xv, kw, vw, k_dyn, v_dyn, cache, table, seq_idx, qsl, seq_lens_t,
            kvc, kv_slot_d, ko, vo, bs, write_state, Q_CLIP, RMS_EPS,
        )  # fmt: skip
        if not write_state:
            _assert_cache(cache, cw.cache0, "compute-only call touched the state")
            for x, off in ((xk, ko), (xv, vo)):
                ops.inkling2_conv_state_write(
                    x, cache, table, seq_idx, qsl, seq_lens_t, off, nkv, bs
                )
        torch.accelerator.synchronize()
        assert_unchanged(ins, snap)
        got = kvc.cpu()
        # slots that were not written must still be NaN, written ones match to
        # fp16 rounding
        assert torch.equal(torch.isnan(got), torch.isnan(expected))
        keep = ~torch.isnan(expected)
        torch.testing.assert_close(
            got[keep].double(), expected[keep].double(), rtol=2e-3, atol=2e-3
        )
        _assert_cache(cache, cw.expected_cache(), "conv state")

    run(False)
    if case == "decode":  # one token per request: kv_prep writes the state itself
        run(True)


# ---------------------------------------------------------------------------
# attention worlds
# ---------------------------------------------------------------------------
def scaled_clip_fp16(x, scale, thr, positions, log_scale, n_floor=LOG_N_FLOOR):
    """Mirror of ``Inkling2Attention._scaled_clip`` (fp32 math, fp16 rounding)."""
    y = x.float() * scale.float()
    norm = torch.linalg.vector_norm(y, dim=-1, keepdim=True)
    y = (y * (thr / (norm + 1e-6)).clamp(max=1.0)).to(torch.float16)
    if log_scale:
        tau = 1.0 + LOG_ALPHA * torch.log(
            ((positions + 1).float() / n_floor).clamp(min=1.0)
        )
        y = (y.float() * tau[:, None, None]).to(torch.float16)
    return y


class AttnWorld:
    """Random paged K|V world plus the per-token inputs of ``inkling2_attention``.

    q_lens / seq_lens per request; ``seq_len <= 0`` marks a padded request.
    """

    def __init__(
        self,
        q_lens,
        seq_lens,
        *,
        nkv=8,
        g=3,
        window=WINDOW,
        rel_extent=REL_EXTENT,
        sharp=True,
        seed=0,
        with_gate=True,
        num_pages=None,
        max_blocks=None,
        n_floor=96,
    ):
        gen = torch.Generator().manual_seed(seed)
        D = HEAD_DIM
        self.q_lens, self.seq_lens = list(q_lens), list(seq_lens)
        self.nkv, self.g, self.nh = nkv, g, nkv * g
        self.window, self.rel_extent = window, rel_extent
        self.log_scale = window == 0
        # log-scale floor of the full layers: far below the model's 8192 so that
        # the log scale is not the identity at the tested lengths
        self.n_floor = n_floor
        self.R, self.T = len(q_lens), sum(q_lens)
        self.qsl = [0]
        for ql in q_lens:
            self.qsl.append(self.qsl[-1] + ql)
        pos = []
        for ql, L in zip(q_lens, seq_lens):
            pos += list(range(L - ql, L)) if L > 0 else [0] * ql
        self.positions = torch.tensor(pos, dtype=torch.int64)

        nb = [_ceil_div(max(L, 0), KV_BLOCK) for L in seq_lens]
        self.num_pages = num_pages or sum(nb) + 6
        pool = PagePool(gen, self.num_pages)
        self.table = torch.zeros(
            self.R, max_blocks or max(nb + [1]) + 2, dtype=torch.int32
        )
        u = F.normalize(_randn(gen, nkv, D), dim=-1)
        boundary = window or rel_extent
        self.k_dense: list[torch.Tensor | None] = []
        self.v_dense: list[torch.Tensor | None] = []
        self.cache = torch.full(
            (self.num_pages, nkv, KV_BLOCK, 2 * D), float("nan"), dtype=torch.float16
        )
        for r, (ql, L) in enumerate(zip(q_lens, seq_lens)):
            if L <= 0:
                self.k_dense.append(None)
                self.v_dense.append(None)
                continue
            k = _randn(gen, L, nkv, D)
            if sharp:
                # spikes on both sides of the window / bias boundary of the
                # first query of the request
                s = (L - ql - boundary) % SPIKE_PERIOD
                idx = torch.arange(s, L, SPIKE_PERIOD)
                idx = torch.cat([idx, idx + 1])
                idx = idx[idx < L]
                k[idx] = 12.0 * u + _randn(gen, len(idx), nkv, D, scale=0.3)
            k, v = k.half(), _randn(gen, L, nkv, D).half()
            self.k_dense.append(k)
            self.v_dense.append(v)
            pages = pool.take(nb[r])
            lo_first = max(0, L - ql - window + 1) if window else 0
            for b in range(nb[r]):
                if b < lo_first // KV_BLOCK:
                    continue  # fully outside every window: null block (NaN)
                self.table[r, b] = pages[b]
                lo, hi = b * KV_BLOCK, min(L, (b + 1) * KV_BLOCK)
                self.cache[pages[b], :, : hi - lo, :D] = k[lo:hi].transpose(0, 1)
                self.cache[pages[b], :, : hi - lo, D:] = v[lo:hi].transpose(0, 1)

        T, nh = self.T, self.nh
        mag = torch.tensor([0.6, 1.0, 3.0])[torch.randint(0, 3, (T, nh), generator=gen)]
        noise = F.normalize(_randn(gen, T, nh, D), dim=-1)
        if sharp:
            ug = u.repeat_interleave(g, 0)[None]  # head h -> kv head h // g
            dirs = F.normalize(0.7 * ug + 0.71 * noise, dim=-1)
        else:
            dirs = noise
        self.q_raw = (dirs * mag[..., None] * math.sqrt(D)).half().view(T, nh * D)
        self.rel_raw = (
            (_randn(gen, T, nh, D_REL) * (mag[..., None] * 2.0))
            .half()
            .view(T, nh * D_REL)
        )
        self.sink_raw = (_randn(gen, T, nh) + (5.0 if sharp else 4.0)).half()
        self.gate_raw = _randn(gen, T, nh * D, scale=1.5).half() if with_gate else None
        self.q_scale = 0.7 + 0.6 * _rand(gen, nh, D)
        self.rel_scale = 0.7 + 0.6 * _rand(gen, nh, D_REL)
        # rows past rel_extent are garbage that must never be used
        self.proj_t = _randn(
            gen, rel_extent + 8, D_REL, scale=0.3 if sharp else 0.1
        ).half()
        self.seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32)
        self.qsl_t = torch.tensor(self.qsl, dtype=torch.int32)

    # -- oracle ------------------------------------------------------------
    def prepped_q(self):
        return scaled_clip_fp16(
            self.q_raw.view(self.T, self.nh, HEAD_DIM),
            self.q_scale,
            Q_CLIP,
            self.positions,
            self.log_scale,
            self.n_floor,
        )

    def prepped_rel(self):
        return scaled_clip_fp16(
            self.rel_raw.view(self.T, self.nh, D_REL),
            self.rel_scale,
            REL_CLIP,
            self.positions,
            self.log_scale,
            self.n_floor,
        )

    def oracle_core(self) -> torch.Tensor:
        """Pre-gate attention output [T, nh, D] in float64."""
        R = self.rel_extent
        q = self.prepped_q().double()
        rel = self.prepped_rel().double()
        sink = self.sink_raw.double()
        proj = self.proj_t[:R].double()
        out = torch.zeros(self.T, self.nh, HEAD_DIM, dtype=torch.float64)
        for r, (ql, L) in enumerate(zip(self.q_lens, self.seq_lens)):
            if L <= 0 or ql == 0:
                continue
            s, e = self.qsl[r], self.qsl[r + 1]
            dist = torch.arange(L - ql, L)[:, None] - torch.arange(L)[None, :]
            vis = dist >= 0
            if self.window:
                vis &= dist < self.window
            rel_logits = torch.einsum("thd,rd->thr", rel[s:e], proj)  # [ql, nh, R]
            idx = dist.clamp(0, R - 1)
            kd, vd = self.k_dense[r], self.v_dense[r]
            assert kd is not None and vd is not None
            for kvh in range(self.nkv):
                kk = kd[:, kvh].double()
                vv = vd[:, kvh].double()
                for gi in range(self.g):
                    h = kvh * self.g + gi
                    sc = (q[s:e, h] @ kk.T) * HEAD_DIM**-0.5  # [ql, L]
                    bias = torch.gather(rel_logits[:, h], 1, idx)
                    bias = bias.masked_fill((dist < 0) | (dist >= R), 0.0)
                    sc = (sc + bias).masked_fill(~vis, float("-inf"))
                    p = torch.softmax(torch.cat([sc, sink[s:e, h, None]], -1), -1)
                    out[s:e, h] = p[:, :-1] @ vv
        return out

    def oracle(self) -> torch.Tensor:
        o = self.oracle_core()
        if self.gate_raw is not None:
            o = o * torch.sigmoid(self.gate_raw.double().view(self.T, self.nh, -1))
        return o.view(self.T, self.nh * HEAD_DIM)

    # -- device inputs -----------------------------------------------------
    def device_args(self, *, strided=False, token_major=True, layers=1):
        """Tensors of the op; ``strided`` slices q / gate / rel / sink out of one
        wider buffer (a fused projection output)."""
        dev = "cuda"
        T = self.T
        parts = {"q": self.q_raw, "gate": self.gate_raw, "rel": self.rel_raw}
        parts["sink"] = self.sink_raw
        if strided:
            parts = {k: v for k, v in parts.items() if v is not None}
            width = _ceil_div(sum(v.shape[1] for v in parts.values()), 8) * 8 + 8
            buf = torch.full((T, width), float("nan"), dtype=torch.float16)
            col, views = 0, {}
            for name, v in parts.items():
                buf[:, col : col + v.shape[1]] = v
                views[name] = (col, v.shape[1])
                col += v.shape[1]
            buf = buf.to(dev)
            dev_parts = {n: buf[:, c : c + w] for n, (c, w) in views.items()}
            dev_parts.setdefault("gate", None)
        else:
            dev_parts = {
                n: (v.to(dev) if v is not None else None) for n, v in parts.items()
            }
        return dict(
            **dev_parts,
            q_scale=self.q_scale.to(dev),
            rel_scale=self.rel_scale.to(dev),
            rel_proj_t=self.proj_t.to(dev),
            positions=self.positions.to(dev),
            kv_cache=to_device_cache(self.cache, token_major, layers),
            block_table=self.table.to(dev),
            seq_lens=self.seq_lens_t.to(dev),
            query_start_loc=self.qsl_t.to(dev),
        )


def launch_attention(w: AttnWorld, a, out, ws, *, splits, max_query_len):
    _ops().inkling2_attention(
        a["q"], a["rel"], a["sink"], a["gate"], a["q_scale"], a["rel_scale"],
        a["rel_proj_t"], a["positions"], a["kv_cache"], a["block_table"],
        a["seq_lens"], a["query_start_loc"], out, ws,
        w.window, w.rel_extent, w.log_scale, LOG_ALPHA, float(w.n_floor),
        Q_CLIP, REL_CLIP, max_query_len, splits[0], splits[1],
    )  # fmt: skip


def run_attention(
    w: AttnWorld,
    *,
    splits=None,
    strided=False,
    pos_int32=False,
    token_major=True,
    layers=1,
):
    a = w.device_args(strided=strided, token_major=token_major, layers=layers)
    if pos_int32:
        a["positions"] = a["positions"].to(torch.int32)
    out = torch.full(
        (w.T, w.nh * HEAD_DIM), float("nan"), dtype=torch.float16, device="cuda"
    )
    max_q = max(w.q_lens)
    if splits is None:  # the shipped policy
        splits = _ops().inkling2_attention_splits(
            w.R, w.nkv, w.window, max(w.seq_lens), max_q
        )
    # exactly sized and NaN-poisoned: a partial row read before it is written,
    # or two requests sharing a row, shows up in the output
    ws = torch.full(
        (_ops().inkling2_attention_workspace_numel(w.R, w.T, w.nh, *splits),),
        float("nan"),
        dtype=torch.float32,
        device="cuda",
    )
    launch_attention(w, a, out, ws, splits=splits, max_query_len=max_q)
    torch.accelerator.synchronize()
    return out


def check_attention(out, ref, what):
    got = out.cpu().double()
    assert torch.isfinite(got).all(), (
        f"{what}: non-finite output (unwritten row or a read of poisoned memory)"
    )
    err = (got - ref).abs()
    bound = ATTN_ATOL + ATTN_RTOL * ref.abs()
    worst = (err / bound).max().item()
    rel_err = (err / ref.abs().clamp(min=1e-2)).max().item()
    print(
        f"inkling2-attn {what}: max_abs_err={err.max().item():.3e} "
        f"max_rel_err(|ref|>=1e-2)={rel_err:.3e} worst_err/bound={worst:.3f}"
    )
    if worst > 1.0:
        t, c = divmod(int((err / bound).argmax()), got.shape[1])
        raise AssertionError(
            f"{what}: err {err.max().item():.3e} exceeds the bound (token {t}, "
            f"head {c // HEAD_DIM}, dim {c % HEAD_DIM}): got {got[t, c].item():.5f} "
            f"ref {ref[t, c].item():.5f}"
        )


# decode: one launch with every interesting length (page 63/64/65, window
# 1023/1024/1025, longer contexts)
DECODE_LENS = [1, 2, 3, 63, 64, 65, 127, 129, 1023, 1024, 1025, 1088, 2049, 4999]
# prefill / chunked-prefill requests (q_len, seq_len)
PREFILL_REQS = [
    (3, 3),
    (64, 64),
    (257, 257),
    (1, 1),
    (16, 16),
    (17, 80),
    (64, 1088),  # the first query sits exactly one window from position 0
    (257, 1300),
    (3, 1025),
    (64, 5000),
    (31, 1023),
    (33, 1024),
]
MIXED_REQS = [
    (1, 1024),
    (64, 1088),
    (1, 65),
    (257, 1281),
    (3, 3),
    (1, 0),  # padded request
    (0, 17),  # empty request
    (1, 1025),
    (16, 2049),
    (1, 4999),
]


def _world(reqs, **kw):
    return AttnWorld([q for q, _ in reqs], [s for _, s in reqs], **kw)


@requires_kernels
@pytest.mark.parametrize("window", [WINDOW, 0], ids=["sliding", "full"])
@pytest.mark.parametrize("nkv", [8, 4, 2])
@pytest.mark.parametrize("token_major", [True, False], ids=["bhc", "hbc"])
def test_attention_decode(window, nkv, token_major):
    w = AttnWorld([1] * len(DECODE_LENS), DECODE_LENS, nkv=nkv, window=window, seed=1)
    out = run_attention(w, token_major=token_major)
    check_attention(out, w.oracle(), f"decode nkv={nkv} window={window}")


@requires_kernels
@pytest.mark.parametrize("window", [WINDOW, 0], ids=["sliding", "full"])
@pytest.mark.parametrize("nkv", [8, 4, 2])
@pytest.mark.parametrize("token_major", [True, False], ids=["bhc", "hbc"])
def test_attention_prefill(window, nkv, token_major):
    w = _world(PREFILL_REQS, nkv=nkv, window=window, seed=2)
    out = run_attention(w, token_major=token_major)
    check_attention(out, w.oracle(), f"prefill nkv={nkv} window={window}")


@requires_kernels
@pytest.mark.parametrize("window", [WINDOW, 0], ids=["sliding", "full"])
@pytest.mark.parametrize("token_major", [True, False], ids=["bhc", "hbc"])
def test_attention_block_outermost_layers(window, token_major):
    """Engine ``BL*`` layouts: other layers' pages sit between this layer's blocks."""
    w = _world(MIXED_REQS, nkv=8, window=window, seed=13)
    out = run_attention(w, token_major=token_major, layers=3)
    check_attention(out, w.oracle(), f"block-outermost layers=3 window={window}")


@requires_kernels
@pytest.mark.parametrize("window", [WINDOW, 0], ids=["sliding", "full"])
@pytest.mark.parametrize("splits", [(1, 1), (4, 2), (32, 5), (256, 16), None])
def test_attention_mixed_splits(window, splits):
    w = _world(MIXED_REQS, nkv=8, window=window, seed=3)
    out = run_attention(w, splits=splits)
    check_attention(out, w.oracle(), f"mixed window={window} splits={splits}")


@requires_kernels
@pytest.mark.parametrize("g", [1, 2, 4])
@pytest.mark.parametrize("window", [WINDOW, 0], ids=["sliding", "full"])
def test_attention_group_sizes(g, window):
    w = _world(MIXED_REQS, nkv=2, g=g, window=window, seed=4)
    check_attention(
        run_attention(w, splits=(3, 2)), w.oracle(), f"group g={g} window={window}"
    )


@requires_kernels
@pytest.mark.parametrize("window", [WINDOW, 0], ids=["sliding", "full"])
def test_attention_unsharp_no_gate_strided(window):
    # plain random data, no gate, q / rel / sink as column slices of one fused
    # buffer, int32 positions
    w = _world(MIXED_REQS, nkv=4, window=window, sharp=False, with_gate=False, seed=5)
    out = run_attention(w, strided=True, pos_int32=True)
    check_attention(out, w.oracle(), f"unsharp window={window}")


@requires_kernels
def test_attention_gate_strided():
    w = _world(MIXED_REQS, nkv=4, window=WINDOW, seed=15)
    out = run_attention(w, splits=(2, 3), strided=True)
    check_attention(out, w.oracle(), "gate in the fused buffer")


@requires_kernels
def test_attention_small_rel_extent():
    # the relative bias is shorter than the window: dist >= rel_extent adds 0
    w = _world(MIXED_REQS, nkv=4, window=WINDOW, rel_extent=200, seed=6)
    check_attention(run_attention(w, splits=(2, 2)), w.oracle(), "rel_extent=200")


@requires_kernels
def test_attention_padded_rows_are_defined():
    w = AttnWorld([1, 1, 1], [0, 5, 0], nkv=2, seed=7)
    out = run_attention(w)
    got = out.cpu().float()
    assert torch.count_nonzero(got[0]) == 0 and torch.count_nonzero(got[2]) == 0
    check_attention(out, w.oracle(), "padded rows")


@requires_kernels
def test_attention_checks_splits_and_workspace():
    w = _world(MIXED_REQS, nkv=8, window=WINDOW, seed=3)
    a = w.device_args()
    out = torch.zeros(w.T, w.nh * HEAD_DIM, dtype=torch.float16, device="cuda")
    max_q = max(w.q_lens)
    splits = (4, 2)
    need = _ops().inkling2_attention_workspace_numel(w.R, w.T, w.nh, *splits)
    short = torch.zeros(need - 1, dtype=torch.float32, device="cuda")
    with pytest.raises(RuntimeError, match="workspace"):
        launch_attention(w, a, out, short, splits=splits, max_query_len=max_q)
    exact = torch.zeros(need, dtype=torch.float32, device="cuda")
    for bad in ((0, 1), (1, 0), (257, 1), (1, 257)):
        with pytest.raises(RuntimeError, match="splits"):
            launch_attention(w, a, out, exact, splits=bad, max_query_len=max_q)
    launch_attention(w, a, out, exact, splits=splits, max_query_len=max_q)
    torch.accelerator.synchronize()


@requires_kernels
def test_attention_leaves_its_inputs_alone():
    """Only ``out`` and the workspace are written: every other argument,
    including the paged cache, comes back bit-identical."""
    w = _world(MIXED_REQS, nkv=8, window=0, seed=3)
    a = w.device_args()
    splits = (4, 2)
    out = torch.zeros(w.T, w.nh * HEAD_DIM, dtype=torch.float16, device="cuda")
    ws = torch.zeros(
        _ops().inkling2_attention_workspace_numel(w.R, w.T, w.nh, *splits),
        dtype=torch.float32,
        device="cuda",
    )
    ins = list(a.values())
    snap = [None if t is None else t.clone() for t in ins]
    launch_attention(w, a, out, ws, splits=splits, max_query_len=max(w.q_lens))
    torch.accelerator.synchronize()
    assert_unchanged(ins, snap)


@requires_kernels
def test_ops_trace_under_fake_tensors():
    """torch.compile traces the ops through their registered fake impls."""
    from torch._subclasses.fake_tensor import FakeTensorMode

    ops = _ops()
    w = _world(MIXED_REQS, nkv=8, window=WINDOW, seed=3)
    real = w.device_args()
    mode = FakeTensorMode()

    def fake(*shape, dtype=torch.float32):
        return torch.empty(*shape, dtype=dtype, device="cuda")

    i32, i64 = torch.int32, torch.int64
    T, C, nkv = 8, 1024, 8
    with mode:
        a = {k: None if v is None else mode.from_tensor(v) for k, v in real.items()}
        out = fake(w.T, w.nh * HEAD_DIM, dtype=torch.float16)
        ws = fake(ops.inkling2_attention_workspace_numel(w.R, w.T, w.nh, 4, 2))
        launch_attention(w, a, out, ws, splits=(4, 2), max_query_len=max(w.q_lens))
        meta = [fake(2, 4, dtype=i32), fake(T, dtype=i32)]  # block table, seq_idx
        meta += [fake(3, dtype=i32), fake(2, dtype=i32)]  # query_start_loc, seq_lens
        conv = fake(4, nkv, 3, C, dtype=torch.float16)
        ops.inkling2_sconv_static(
            fake(T, C), fake(C, 1, 4), fake(T, C), conv, *meta, 0, nkv, 64, False
        )
        ops.inkling2_sconv_dynamic(
            fake(T, C), fake(4, C), fake(T, nkv * 4, dtype=torch.float16),
            fake(T, C), conv, *meta, 0, nkv, 64, False,
        )  # fmt: skip
        ops.inkling2_conv_state_write(
            fake(T, C, dtype=torch.float16), conv, *meta, 0, nkv, 64
        )
        kv = fake(4, nkv, 64, 2 * HEAD_DIM, dtype=torch.float16)
        ops.inkling2_kv_prep(
            fake(T, C, dtype=torch.float16), fake(T, C, dtype=torch.float16),
            fake(C, 1, 4), fake(C, 1, 4), None, None, conv, meta[0], meta[1],
            meta[2], meta[3], kv, fake(T, dtype=i64), 0, 128, 64, False, Q_CLIP,
            RMS_EPS,
        )  # fmt: skip


@requires_kernels
@pytest.mark.parametrize("splits", [None, (16, 8), (128, 4)])
def test_attention_million_tokens_full_layer(splits):
    """TP=4 full layer (2 kv heads) at the model's 1,048,576 context."""
    reqs = [(1, 1048576), (1, 1000003), (3, 1048576), (1, 70001)]
    w = _world(reqs, nkv=2, window=0, n_floor=LOG_N_FLOOR, seed=12)
    out = run_attention(w, splits=splits)
    check_attention(out, w.oracle(), f"1M tokens full layer splits={splits}")


@requires_kernels
def test_attention_cuda_graph_replay():
    """Metadata is read on device: replay a captured launch with new lengths."""
    reqs_a = [(1, 700), (1, 1025), (16, 600), (1, 70), (33, 2000)]
    reqs_b = [(1, 1100), (1, 33), (16, 1500), (1, 4000), (33, 90)]
    shape = dict(nkv=8, num_pages=140, max_blocks=70)
    wa = _world(reqs_a, seed=8, **shape)
    wb = _world(reqs_b, seed=9, **shape)
    splits = (4, 2)
    a = wa.device_args()
    out = torch.full(
        (wa.T, wa.nh * HEAD_DIM), float("nan"), dtype=torch.float16, device="cuda"
    )
    ws = torch.empty(
        _ops().inkling2_attention_workspace_numel(wa.R, wa.T, wa.nh, *splits),
        dtype=torch.float32,
        device="cuda",
    )

    def launch():
        launch_attention(wa, a, out, ws, splits=splits, max_query_len=33)

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        launch()  # warm-up outside the capture
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        launch()
    for name, new in wb.device_args().items():
        a[name].copy_(new)
    out.fill_(float("nan"))
    graph.replay()
    torch.accelerator.synchronize()
    check_attention(out, wb.oracle(), "graph replay with new lengths / block table")


# ---------------------------------------------------------------------------
# oracle vs the repo's eager torch attention core (CPU)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("sliding", [True, False], ids=["sliding", "full"])
def test_scaled_clip_matches_repo(sliding):
    from vllm.models.inkling2_loop.amd.attention import Inkling2Attention

    gen = torch.Generator().manual_seed(21)
    T, nh = 37, 6
    mag = torch.tensor([0.3, 1.0, 4.0])[torch.randint(0, 3, (T, nh, 1), generator=gen)]
    positions = torch.randint(0, 40000, (T,), generator=gen)
    for d, thr in ((HEAD_DIM, Q_CLIP), (D_REL, REL_CLIP)):
        x = (_randn(gen, T, nh, d, scale=3.0) * mag).half()
        scale = 0.7 + 0.6 * _rand(gen, nh, d)
        me = SimpleNamespace(
            is_sliding=sliding,
            config=SimpleNamespace(
                log_scaling_alpha=LOG_ALPHA, log_scaling_n_floor=LOG_N_FLOOR
            ),
        )
        ref = Inkling2Attention._scaled_clip(me, x, scale, thr, positions)
        mine = scaled_clip_fp16(x, scale, thr, positions, not sliding, LOG_N_FLOOR)
        assert torch.equal(ref, mine)


@pytest.mark.parametrize("window", [32, 0])
def test_oracle_matches_repo_attend(window, monkeypatch):
    from vllm.models.inkling2_loop.amd import attention as attn_mod

    w = AttnWorld(
        [5, 1, 3, 40], [40, 70, 3, 40], nkv=2, g=3, window=window,
        rel_extent=32, sharp=False, seed=11,
    )  # fmt: skip

    # the repo's _attend reads K/V through kv_read; serve the dense world
    def fake_kv_read(owner, md, r, lo, total):
        return w.k_dense[r][lo:total], w.v_dense[r][lo:total]

    monkeypatch.setattr(attn_mod, "kv_read", fake_kv_read)
    me = SimpleNamespace(
        num_kv_heads=w.nkv,
        groups=w.g,
        window=window or None,
        rel_extent=w.rel_extent,
        scaling=HEAD_DIM**-0.5,
    )
    md = SimpleNamespace(query_start_loc=w.qsl_t, seq_lens=w.seq_lens_t, num_reqs=w.R)
    rel_logits = torch.einsum(
        "thd,rd->thr", w.prepped_rel().float(), w.proj_t[: w.rel_extent].float()
    )
    ref = attn_mod.Inkling2Attention._attend(
        me, w.prepped_q(), rel_logits, w.sink_raw.float(), w.positions, None, md
    )
    torch.testing.assert_close(w.oracle_core().float(), ref, rtol=2e-4, atol=2e-5)


# ---------------------------------------------------------------------------
# split policy helpers (pure Python)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("window", [WINDOW, 0], ids=["sliding", "full"])
def test_split_policy_properties(window):
    split_fn = _ops().inkling2_attention_splits
    for nkv in (2, 4, 8):
        for ctx in (1, 17, 1024, 16384, 131072, 1048576):
            last_a = None
            for reqs in (1, 2, 4, 8, 16, 32, 128, 1024):
                for max_q in (1, 2, 16, 17, 64, 1024):
                    s_a, s_b = split_fn(reqs, nkv, window, ctx, max_q)
                    assert 1 <= s_a <= 256 and 1 <= s_b <= 16
                    if max_q == 1:  # pure decode: no tile kernel, no tile partials
                        assert s_b == 1
                s_a = split_fn(reqs, nkv, window, ctx, 1)[0]
                if last_a is not None:  # more requests never ask for more splits
                    assert s_a <= last_a
                last_a = s_a
    for nkv in (2, 8):
        for reqs in (1, 4, 64):
            by_ctx = [
                split_fn(reqs, nkv, window, c, 1)[0]
                for c in (1, 1024, 16384, 131072, 1048576)
            ]
            assert by_ctx == sorted(by_ctx)  # a longer context never asks for fewer
            if window:  # a sliding layer only ever attends its window
                assert len(set(by_ctx[1:])) == 1


def test_split_policy_saturates_the_gpu_at_decode():
    """One request on a long full layer gets 256 waves, a big batch gets one split."""
    split_fn = _ops().inkling2_attention_splits
    assert split_fn(1, 8, 0, 131072, 1)[0] * 8 == 256
    assert split_fn(1, 2, 0, 1048576, 1)[0] * 2 == 256
    assert split_fn(64, 8, 0, 131072, 1)[0] == 1
    assert split_fn(1, 8, WINDOW, 131072, 1)[0] * 8 == 256


def test_workspace_numel_formula():
    numel = _ops().inkling2_attention_workspace_numel
    assert numel(3, 20, 24, 4, 2) == (4 * 3 + 2 * 20) * 24 * 130
    assert numel(5, 5, 6, 1, 1) == (5 + 5) * 6 * 130
