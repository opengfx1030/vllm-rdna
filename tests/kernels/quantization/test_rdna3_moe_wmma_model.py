# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU model of ``moe_gptq_gemm_rdna3_wmma`` (csrc/rocm/moe_q_gemm_rdna3_wmma.cu).

The kernel only runs on gfx1100, so this file re-executes its schedule in torch,
index for index: the (token block, N tile) grid, the wave/fragment column
assignment with the N-tail clamp, the activation gather into double-buffered
LDS chunks, the per-lane packed-word addressing with the one-sub-step-ahead
weight prefetch, the group scale/zero refresh pipeline, the precise W4 dequant,
the v_wmma_f32_16x16x16 fragment layout and the output scatter with the Q32
fixed-point rule. The result is compared with a dense fp64 reference built from
the packed bytes alone, and the schedule is mutation-tested: every deliberate
index error in ``MUTANTS`` has to be caught.

What this validates: the mapping DESIGN (every index expression the kernel's
header documents) and its arithmetic, against the contract of the op. What it
cannot validate: the compiled code (barrier placement, register allocation,
hardware WMMA rounding), which is the job of test_rdna3_moe_wmma.py on a
gfx1100 device. ``test_kernel_source_still_contains_the_modelled_index_math``
pins the index expressions of the kernel source, so an edit of the kernel that
changes the mapping fails here until this model is updated with it.

The helpers without a ``test_`` prefix (packing, routing, dense reference) are
shared with the GPU test.
"""

import types
from dataclasses import dataclass
from pathlib import Path

import pytest
import regex as re
import torch

KERNEL_SOURCE = (
    Path(__file__).resolve().parents[3] / "csrc" / "rocm" / "moe_q_gemm_rdna3_wmma.cu"
)

LANES = 32
Q32_SCALE = float(1 << 32)
Q32_RANGE = float(1 << 30)


@pytest.fixture(autouse=True, scope="module")
def _single_torch_thread():
    """The emulation is thousands of tiny batched ops; a pool of dozens of
    threads spends its time synchronizing (100x slower on a busy many-core
    host)."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


# ---------------------------------------------------------------------------
# Tile shapes (parsed from the kernel source)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Tile:
    block_m: int
    waves: int
    nf: int
    kc: int

    @property
    def mt(self) -> int:
        return self.block_m // 16

    @property
    def threads(self) -> int:
        return self.waves * LANES

    @property
    def tile_n(self) -> int:
        return self.waves * self.nf * 16

    @property
    def subs(self) -> int:
        return self.kc // 32

    @property
    def vec_per_row(self) -> int:
        return self.kc // 8

    @property
    def rows_per_pass(self) -> int:
        return self.threads // self.vec_per_row

    @property
    def passes(self) -> int:
        return self.block_m // self.rows_per_pass

    @property
    def lds_row_vec(self) -> int:
        return self.vec_per_row + 1

    @property
    def lds_bytes(self) -> int:
        return 2 * self.block_m * self.lds_row_vec * 16


TILES = {
    16: Tile(16, waves=4, nf=4, kc=128),
    32: Tile(32, waves=8, nf=2, kc=128),
    64: Tile(64, waves=8, nf=2, kc=64),
}
BLOCK_MS = sorted(TILES)

_TILE_STRUCT = re.compile(
    r"struct\s+Tile<(\d+)>\s*\{\s*static\s+constexpr\s+int\s+"
    r"WAVES\s*=\s*(\d+)\s*,\s*NF\s*=\s*(\d+)\s*,\s*KC\s*=\s*(\d+)\s*;\s*\}\s*;"
)


def parse_kernel_tiles(source: str) -> dict[int, Tile]:
    return {
        int(m[1]): Tile(int(m[1]), waves=int(m[2]), nf=int(m[3]), kc=int(m[4]))
        for m in _TILE_STRUCT.finditer(source)
    }


# ---------------------------------------------------------------------------
# Weights: GPTQ packing, the documented format, and the dense decode
# ---------------------------------------------------------------------------


@dataclass
class Weights:
    q: torch.Tensor  # [E, K, N] int64, 4-bit values
    zeros: torch.Tensor  # [E, G, N] int64, stored zero (effective zero - 1)
    scales: torch.Tensor  # [E, G, N] fp16/bf16
    b_q: torch.Tensor  # [E, K/8, N] int32, GPTQ-shuffled
    b_z: torch.Tensor  # [E, G, N/8] int32, nibble n % 8 of word n / 8

    @property
    def group_size(self) -> int:
        return self.q.shape[1] // self.scales.shape[1]


def to_int32(words: torch.Tensor) -> torch.Tensor:
    """uint32 values held in int64 -> int32 with two's complement wrap."""
    return ((words + (1 << 31)) % (1 << 32) - (1 << 31)).to(torch.int32)


def pack_k_sequential(q: torch.Tensor) -> torch.Tensor:
    """[E, K, N] nibbles -> [E, K/8, N] words, K = 8 kw + i in nibble i."""
    E, K, N = q.shape
    qv = q.reshape(E, K // 8, 8, N)
    words = torch.zeros(E, K // 8, N, dtype=torch.int64)
    for i in range(8):
        words |= qv[:, :, i, :] << (4 * i)
    return words


def gptq_shuffle(words: torch.Tensor) -> torch.Tensor:
    """Literal port of shuffle_4bit_8 (qdq_4_rdna3.cuh): even K to nibbles 0..3,
    odd K to nibbles 4..7."""
    qa = words.clone()
    qb = torch.zeros_like(words)
    for i in range(4):
        qa0 = qa & 0x0F
        qa1 = (qa & 0xF0) >> 4
        qa = qa >> 8
        qb |= qa1 << (i * 4 + 16)
        qb |= qa0 << (i * 4)
    return qb


def pack_zeros(zeros: torch.Tensor) -> torch.Tensor:
    """[E, G, N] -> [E, G, N/8] words, column n in nibble n % 8 of word n / 8."""
    E, G, N = zeros.shape
    zv = zeros.reshape(E, G, N // 8, 8)
    words = torch.zeros(E, G, N // 8, dtype=torch.int64)
    for i in range(8):
        words |= zv[..., i] << (4 * i)
    return to_int32(words)


def make_weights(E, K, N, group_size, dtype, gen, scale_range=(0.01, 0.1)) -> Weights:
    groups = K // group_size
    q = torch.randint(0, 16, (E, K, N), generator=gen, dtype=torch.int64)
    # Every stored zero 0..15 occurs, so every effective zero 1..16 is exercised.
    zeros = torch.randint(0, 16, (E, groups, N), generator=gen, dtype=torch.int64)
    lo, hi = scale_range
    scales = (torch.rand(E, groups, N, generator=gen) * (hi - lo) + lo).to(dtype)
    return Weights(
        q=q,
        zeros=zeros,
        scales=scales,
        b_q=to_int32(gptq_shuffle(pack_k_sequential(q))),
        b_z=pack_zeros(zeros),
    )


# The documented slot order of a packed word: K = 8 kw + s sits in nibble
# NIBBLE_OF_SLOT[s] (kernel header, "Weights (B)").
NIBBLE_OF_SLOT = [0, 4, 1, 5, 2, 6, 3, 7]


def dequantize_expert(b_q, b_z, scales, e: int) -> torch.Tensor:
    """fp64 [K, N] weights of expert ``e`` decoded from the PACKED tensors alone
    (the format documented in the kernel header), unrounded:
    (nibble - zero) * scale. Works on any device."""
    kw, N = b_q.shape[1:]
    K = kw * 8
    groups = scales.shape[1]
    words = b_q[e].long() & 0xFFFFFFFF
    nibbles = torch.empty(K, N, dtype=torch.int64, device=words.device)
    for s in range(8):
        nibbles[s::8, :] = (words >> (4 * NIBBLE_OF_SLOT[s])) & 0xF
    zwords = b_z[e].long() & 0xFFFFFFFF
    zeros = torch.empty(groups, N, dtype=torch.int64, device=words.device)
    for i in range(8):
        zeros[:, i::8] = (zwords >> (4 * i)) & 0xF
    gs = K // groups
    zeros_k = zeros.repeat_interleave(gs, dim=0) + 1  # GPTQ v1
    scales_k = scales[e].double().repeat_interleave(gs, dim=0)
    return (nibbles - zeros_k).double() * scales_k


def dequantize_packed(w: Weights) -> torch.Tensor:
    """fp64 [E, K, N] weights of every expert, see dequantize_expert."""
    return torch.stack(
        [dequantize_expert(w.b_q, w.b_z, w.scales, e) for e in range(w.b_q.shape[0])]
    )


def rounded_weights(w: Weights, dtype) -> torch.Tensor:
    """fp64 view of the weights as the kernel feeds them to the matrix unit:
    (nibble - zero) * scale rounded ONCE to the activation dtype."""
    return dequantize_packed(w).to(dtype).double()


# ---------------------------------------------------------------------------
# Routing (a pure-torch moe_align_block_size) and the contract oracle
# ---------------------------------------------------------------------------


@dataclass
class Routing:
    sorted_ids: torch.Tensor  # int32 [blocks * block_m], sentinel = num_pairs
    expert_ids: torch.Tensor  # int32 [blocks]
    ntpp: torch.Tensor  # int32 [1] num_tokens_post_padded
    block_m: int
    num_pairs: int


def make_topk_ids(M, top_k, num_experts, gen, style="random") -> torch.Tensor:
    if style == "random":  # distinct experts per token
        return torch.stack(
            [torch.randperm(num_experts, generator=gen)[:top_k] for _ in range(M)]
        )
    if style == "one_expert":  # every pair of the batch goes to expert 0
        return torch.zeros(M, top_k, dtype=torch.int64)
    if style == "sparse":  # only the first two experts are ever used
        return torch.randint(0, 2, (M, top_k), generator=gen)
    raise ValueError(style)


def align_block_size(
    topk_ids, num_experts, block_m, *, drop_experts=(), stale_blocks=0
) -> Routing:
    """Pairs sorted by expert, each expert padded to whole blocks with the
    sentinel ``num_pairs``. ``drop_experts``: their blocks carry expert id -1
    (an expert of another rank). ``stale_blocks``: capacity blocks behind
    num_tokens_post_padded that still hold valid-looking pair ids and a valid
    expert id, as the left-overs of a previous call would."""
    flat = topk_ids.reshape(-1)
    pairs = flat.numel()
    ids, experts = [], []
    for e in range(num_experts):
        sel = (flat == e).nonzero().flatten()
        if sel.numel() == 0:
            continue
        nblk = -(-sel.numel() // block_m)
        chunk = torch.full((nblk * block_m,), pairs, dtype=torch.int32)
        chunk[: sel.numel()] = sel.to(torch.int32)
        ids.append(chunk)
        experts += [-1 if e in drop_experts else e] * nblk
    used = sum(c.numel() for c in ids)
    for s in range(stale_blocks):
        ids.append((torch.arange(block_m, dtype=torch.int32) + s) % max(pairs, 1))
        experts.append(min(1, num_experts - 1))
    return Routing(
        sorted_ids=torch.cat(ids) if ids else torch.empty(0, dtype=torch.int32),
        expert_ids=torch.tensor(experts, dtype=torch.int32),
        ntpp=torch.tensor([used], dtype=torch.int32),
        block_m=block_m,
        num_pairs=pairs,
    )


def scheduled_rows(routing: Routing, top_k: int, size_m: int, num_experts: int):
    """(pair, expert) of every row of c the op contract defines: rows of blocks
    that are neither past num_tokens_post_padded nor expert -1, minus padding."""
    bm = routing.block_m
    limit = int(routing.ntpp[0])
    rows = []
    for tb in range(routing.sorted_ids.numel() // bm):
        if tb * bm >= limit:
            break
        e = int(routing.expert_ids[tb])
        if e < 0 or e >= num_experts:
            continue
        for r in range(bm):
            pair = int(routing.sorted_ids[tb * bm + r])
            if pair >= 0 and pair // top_k < size_m:
                rows.append((pair, e))
    return rows


def reference_pair_values(a, w_rounded, rows, top_k):
    """fp64 [S, N] pair values and the [S, N] sum of |terms| (the rounding
    error scale), from the dequantized weights; rows as in scheduled_rows."""
    N = w_rounded.shape[2]
    values = torch.zeros(len(rows), N, dtype=torch.float64)
    bound = torch.zeros(len(rows), N, dtype=torch.float64)
    experts = torch.tensor([e for _, e in rows], dtype=torch.int64)
    tokens = torch.tensor([p // top_k for p, _ in rows], dtype=torch.int64)
    for e in experts.unique().tolist():
        sel = (experts == e).nonzero().flatten()
        act = a[tokens[sel]].double()
        values[sel] = act @ w_rounded[e]
        bound[sel] = act.abs().nan_to_num(posinf=0.0) @ w_rounded[e].abs()
    return values, bound


def q32_finalize(accumulator: torch.Tensor) -> torch.Tensor:
    """int64 [rows, N + 1] -> fp16 [rows, N]: int64 -> fp32 -> fp16, NaN for a
    flagged row and for an fp16 overflow (moe_fixedpoint_to_half_rdna3)."""
    value = (accumulator[:, :-1].float() * (1.0 / (1 << 32))).half()
    poisoned = (accumulator[:, -1:] != 0) | ~torch.isfinite(value)
    return torch.where(poisoned, torch.full_like(value, float("nan")), value)


# ---------------------------------------------------------------------------
# Problem builder
# ---------------------------------------------------------------------------


@dataclass
class Case:
    a: torch.Tensor  # [rows of a, K]
    w: Weights
    topk_w: torch.Tensor | None  # [pairs] fp32, None: no weight multiply
    routing: Routing
    top_k: int  # the op's top_k (1 for the second GEMM)
    output_topk: int  # 0: direct store / one Q32 row per pair
    dtype: torch.dtype

    @property
    def pairs(self) -> int:
        return self.a.shape[0] * self.top_k

    @property
    def out_rows(self) -> int:
        if self.output_topk > 0:
            return -(-self.pairs // self.output_topk)
        return self.pairs


def build_case(
    *,
    seed=0,
    E=4,
    K=384,
    N=272,
    group_size=128,
    M=13,
    top_k=3,
    dtype=torch.float16,
    block_m=32,
    style="random",
    mul=True,
    gemm2=False,
    drop=(),
    stale_blocks=0,
    hot_expert=None,
    nan_token=None,
) -> Case:
    """``gemm2``: the down projection layout (a = one row per pair, op top_k 1,
    output_topk = the routing top_k) instead of the gate/up layout."""
    gen = torch.Generator().manual_seed(seed)
    w = make_weights(E, K, N, group_size, dtype, gen)
    topk_ids = make_topk_ids(M, top_k, E, gen, style)
    routing = align_block_size(
        topk_ids, E, block_m, drop_experts=drop, stale_blocks=stale_blocks
    )
    rows_a = M * top_k if gemm2 else M
    a = torch.randn(rows_a, K, generator=gen)
    if hot_expert is not None:
        # One expert whose pair values reach the Q32 limit while the others
        # stay far below it.
        w.scales[hot_expert] = torch.full_like(w.scales[hot_expert], 2500.0)
        a = a * 1500.0
    if nan_token is not None:
        a[nan_token, 5] = float("nan")
    topk_w = (torch.rand(M * top_k, generator=gen) + 0.25) if mul else None
    return Case(
        a=a.to(dtype),
        w=w,
        topk_w=topk_w,
        routing=routing,
        top_k=1 if gemm2 else top_k,
        output_topk=top_k if gemm2 else 0,
        dtype=dtype,
    )


# ---------------------------------------------------------------------------
# The kernel model
# ---------------------------------------------------------------------------


class Mapping:
    """Index math of the kernel, one method per expression of the kernel
    source. Tensor in, tensor out, so a method can be applied to every thread,
    lane or block at once. The mutants below override exactly one."""

    def __init__(self, tile: Tile):
        self.t = tile

    # Activation staging: thread tid copies vector `stage_vec` of row
    # `stage_row` in pass p.
    def stage_row(self, tid, p):
        return p * self.t.rows_per_pass + tid // self.t.vec_per_row

    def stage_vec(self, tid):
        return tid % self.t.vec_per_row

    # Columns: first column of fragment j of wave w in N tile ny, and the
    # column a lane loads (clamped so a fragment past N reads the last 16).
    def frag_n0(self, ny, wave, j):
        return ny * self.t.tile_n + (wave * self.t.nf + j) * 16

    def n_load(self, frag_n0, lane, size_n):
        return torch.clamp(frag_n0, max=size_n - 16) + (lane & 15)

    # Weight words: K-word row of word r (0..3) of the 32 K at k; the nibble a
    # slot (0..7 of a word) is stored in; the zero and scale of group g.
    def word_row(self, k, r):
        return (k >> 3) + r

    def nibble(self, slot: int) -> int:
        return slot // 2 if slot % 2 == 0 else 4 + slot // 2

    def zero_of(self, zero_word, n):
        return ((zero_word >> ((n & 7) * 4)) & 0xF) + 1

    def zero_word_index(self, g, n, size_n):
        return g * (size_n >> 3) + (n >> 3)

    def scale_index(self, g, n, size_n):
        return g * size_n + n

    # K schedule of one chunk: K of sub step (pr, second) and the first LDS
    # vector of its A fragments.
    def sub_k(self, k_base, pr, second):
        return k_base + 64 * pr + (32 if second else 0)

    def kvec0(self, pr, second):
        return 8 * pr + (4 if second else 0)

    # A fragment of M tile mt: LDS row of a lane, vector of slot half `half`.
    def a_row(self, mt, lane):
        return 16 * mt + (lane & 15)

    def a_vec(self, kvec0, ks, half):
        return kvec0 + 2 * ks + half

    # D fragment: row of slot i of a lane, column of a lane.
    def d_row(self, mt, i, lane):
        return 16 * mt + 2 * i + (lane >> 4)

    def d_col(self, frag_n0, lane):
        return frag_n0 + (lane & 15)

    # Rows of c.
    def out_row(self, pair, output_topk):
        return pair // output_topk if output_topk > 0 else pair

    def weight_index(self, pair, r):
        return pair


@dataclass
class Emulation:
    out: torch.Tensor  # direct: [pairs, N] dtype, NaN-filled; Q32: int64
    pair_value: torch.Tensor  # [pairs, N] fp32 value before rounding / Q32
    writes: torch.Tensor  # [pairs, N] int, times each element was written


def wmma_f32_16x16x16(a_frag, b_frag, c_frag):
    """v_wmma_f32_16x16x16 on [..., 32, 16] A and B fragments (16 values per
    lane) and [..., 32, 8] fp32 accumulators:
        a[lane][s] = A[m = lane % 16][k = s]   (lanes l and l + 16 identical)
        b[lane][s] = B[k = s][n = lane % 16]   (lanes l and l + 16 identical)
        c[lane][i] = C[m = 2 i + lane // 16][n = lane % 16]
    The sum is formed in fp64 and rounded to fp32 once per instruction."""
    for frag in (a_frag, b_frag):
        lo = frag[..., :16, :].contiguous()
        hi = frag[..., 16:, :].contiguous()
        assert torch.equal(lo.view(torch.int16), hi.view(torch.int16)), (
            "WMMA operand lanes l and l + 16 must hold the same data"
        )
    A = a_frag[..., :16, :].double()
    B = b_frag[..., :16, :].double().transpose(-1, -2)
    lane = torch.arange(LANES)
    i = torch.arange(8)
    m_idx = 2 * i[None, :] + (lane // 16)[:, None]
    n_idx = (lane % 16)[:, None].expand(LANES, 8)
    C = torch.zeros(*c_frag.shape[:-2], 16, 16, dtype=torch.float64)
    C[..., m_idx, n_idx] = c_frag.double()
    D = A @ B + C
    return D[..., m_idx, n_idx].float()


def dequant_b_fragment(words2, zero, scale, nibble_pos, dtype):
    """The B fragment of one 16-K step: [.., 2] packed words (slots 0..7 and
    8..15), [..] effective zero and [..] scale -> [.., 16] values of ``dtype``
    with ONE rounding. fp16: (1024 + nibble) - (1024 + zero) is exact, times the
    scale rounded once. bf16: an exact fp32 fma narrowed once (RNE)."""
    octet = torch.arange(16) // 8
    nib = (words2[..., octet] >> (4 * nibble_pos)) & 0xF
    zero, scale = zero[..., None], scale[..., None]
    if dtype == torch.float16:
        q = (1024 + nib).to(torch.float16)
        return (q - (1024 + zero).to(torch.float16)) * scale
    exact = (nib - zero).to(torch.float32) * scale.to(torch.float32)
    return exact.to(torch.bfloat16)


def emulate_kernel(
    tile: Tile,
    mapping_cls,
    case: Case,
    *,
    q32: bool,
    faults: frozenset = frozenset(),
) -> Emulation:
    """Run the kernel schedule for every (token block, N tile) at once; the
    leading dimension G of every tensor is that grid, W the waves, NF the
    fragments per wave, 32 the lanes."""
    t = tile
    m = mapping_cls(t)
    dtype = case.dtype
    a, w, routing = case.a, case.w, case.routing
    top_k, output_topk = case.top_k, case.output_topk
    size_m, size_k = a.shape
    num_experts, _, size_n = w.b_q.shape
    groups = w.scales.shape[1]
    groupsize = size_k // groups
    bm = t.block_m
    pairs = case.pairs
    assert size_k % 32 == 0 and size_n % 16 == 0 and groupsize % 32 == 0

    out_rows = case.out_rows
    res = Emulation(
        out=(
            torch.zeros(out_rows, size_n + 1, dtype=torch.int64)
            if q32
            else torch.full((pairs, size_n), float("nan"), dtype=dtype)
        ),
        pair_value=torch.full((pairs, size_n), float("nan")),
        writes=torch.zeros(pairs, size_n, dtype=torch.int64),
    )

    # Whole-block early exits.
    ntpp = int(routing.ntpp[0])
    active = []
    for tb in range(routing.sorted_ids.numel() // bm):
        if "ignore_ntpp" not in faults and tb * bm >= ntpp:
            continue
        e = int(routing.expert_ids[tb])
        if "no_expert_check" not in faults and (e < 0 or e >= num_experts):
            continue
        active.append(tb)
    if not active:
        return res
    n_tiles = -(-size_n // t.tile_n)
    g_block = torch.tensor(active).repeat_interleave(n_tiles)
    g_ny = torch.arange(n_tiles).repeat(len(active))
    g_exp = routing.expert_ids[g_block].long()
    G = g_block.numel()
    sorted_ids = routing.sorted_ids.long()

    # Epilogue metadata of tile row r (s_out_row / s_weight).
    r_idx = torch.arange(bm)
    row_pair = sorted_ids[g_block[:, None] * bm + r_idx[None]]  # [G, BM]
    meta_ok = (row_pair >= 0) & (row_pair // top_k < size_m)
    if "no_valid_check_meta" in faults:
        meta_ok = torch.ones_like(meta_ok)
    out_row = torch.where(
        meta_ok, m.out_row(row_pair, output_topk), torch.full_like(row_pair, -1)
    )
    if case.topk_w is None:
        weight = torch.ones(G, bm)
    else:
        widx = m.weight_index(row_pair, r_idx[None]).clamp(0, pairs - 1)
        weight = torch.where(meta_ok, case.topk_w[widx], torch.zeros(()))
    if "weight_dropped" in faults:
        weight = torch.ones(G, bm)

    # Activation staging.
    tids = torch.arange(t.threads)
    st_rows = torch.stack([m.stage_row(tids, p) for p in range(t.passes)])  # [P, T]
    st_vec = m.stage_vec(tids)  # [T]
    st_pair = sorted_ids[g_block[:, None, None] * bm + st_rows[None]]  # [G, P, T]
    st_tok = st_pair // top_k
    st_ok = (st_pair >= 0) & (st_tok < size_m)
    if "no_valid_check_stage" in faults:
        st_ok = st_pair >= 0
    lds = torch.full((G, 2, bm, t.lds_row_vec, 8), float("nan"), dtype=dtype)
    lds_chunk = torch.full((G, 2, bm, t.lds_row_vec), -1, dtype=torch.int64)

    def load_a_chunk(k_base):
        k_rem = size_k - k_base
        want = st_ok & (st_vec * 8 < k_rem)[None, None, :]
        assert bool((st_tok[want] < size_m).all()), "activation row out of range"
        col = k_base + 8 * st_vec[:, None] + torch.arange(8)[None]  # [T, 8]
        tok = st_tok.clamp(0, size_m - 1)
        vals = a[tok[..., None], col.clamp(max=size_k - 1)[None, None]]
        return torch.where(want[..., None], vals, torch.zeros((), dtype=dtype))

    def store_a_chunk(buf, vals, chunk):
        for p in range(t.passes):
            lds[:, buf, st_rows[p], st_vec, :] = vals[:, p]
            lds_chunk[:, buf, st_rows[p], st_vec] = chunk

    lanes = torch.arange(LANES)

    def a_frag(buf, chunk, mt, kvec0, ks):
        row = m.a_row(mt, lanes)
        v0, v1 = m.a_vec(kvec0, ks, 0), m.a_vec(kvec0, ks, 1)
        stale = (lds_chunk[:, buf, row, v0] != chunk) | (
            lds_chunk[:, buf, row, v1] != chunk
        )
        assert not bool(stale.any()), "LDS read of a stale or unwritten vector"
        return torch.cat([lds[:, buf, row, v0, :], lds[:, buf, row, v1, :]], dim=-1)

    # Weight columns of every lane.
    waves = torch.arange(t.waves)[None, :, None]
    frags = torch.arange(t.nf)[None, None, :]
    frag_n0 = m.frag_n0(g_ny[:, None, None], waves, frags)  # [G, W, NF]
    n_load = m.n_load(frag_n0[..., None], lanes, size_n)  # [G, W, NF, 32]
    expert = g_exp[:, None, None, None]

    bq = w.b_q.long() & 0xFFFFFFFF
    zq = (w.b_z.long() & 0xFFFFFFFF).reshape(num_experts, -1)
    sc = w.scales.reshape(num_experts, -1)

    def load_words(k):
        rows = [bq[expert, m.word_row(k, r), n_load] for r in range(4)]
        return torch.stack(rows, dim=-1)  # [G, W, NF, 32, 4]

    st = types.SimpleNamespace(group=0, next_group_k=groupsize)

    def load_group_raw(g):
        g = min(g, groups - 1)
        st.zero_raw = zq[expert, m.zero_word_index(g, n_load, size_n)]
        st.scale_raw = sc[expert, m.scale_index(g, n_load, size_n)]

    def make_prep():
        st.zero = m.zero_of(st.zero_raw, n_load)
        st.scale = st.scale_raw

    nibble_pos = torch.tensor([m.nibble(s) for s in range(8)] * 2)

    def dequant(words2):
        return dequant_b_fragment(words2, st.zero, st.scale, nibble_pos, dtype)

    acc = torch.zeros(G, t.waves, t.nf, t.mt, LANES, 8)

    def sub_step(k, cur, nxt, kvec0, buf, chunk):
        if k >= size_k:  # tail chunk: only whole 32-K steps exist
            return
        boundary = st.next_group_k + (32 if "group_late" in faults else 0)
        if k == boundary:
            st.group += 1
            st.next_group_k += groupsize
            make_prep()
            load_group_raw(st.group + 1)
        nxt["w"] = load_words(min(k + 32, size_k - 32))
        for ks in range(2):
            b_frag = dequant(cur["w"][..., 2 * ks : 2 * ks + 2])
            for mt in range(t.mt):
                af = a_frag(buf, chunk, mt, kvec0, ks)[:, None, None]
                acc[:, :, :, mt] = wmma_f32_16x16x16(af, b_frag, acc[:, :, :, mt])

    # Prologue, then the chunk loop with the one-ahead register prefetch.
    load_group_raw(0)
    make_prep()
    load_group_raw(1)
    words_a: dict = {}
    words_b: dict = {}
    pref = load_a_chunk(0)
    words_a["w"] = load_words(0)
    store_a_chunk(0, pref, 0)
    buf = 0
    for chunk, k_base in enumerate(range(0, size_k, t.kc)):
        has_next = k_base + t.kc < size_k
        if has_next:
            pref = load_a_chunk(k_base + t.kc)
        for pr in range(t.subs // 2):
            sub_step(
                m.sub_k(k_base, pr, False),
                words_a,
                words_b,
                m.kvec0(pr, False),
                buf,
                chunk,
            )
            sub_step(
                m.sub_k(k_base, pr, True),
                words_b,
                words_a,
                m.kvec0(pr, True),
                buf,
                chunk,
            )
        if has_next:
            store_a_chunk(
                buf if "store_same_buffer" in faults else buf ^ 1, pref, chunk + 1
            )
        buf ^= 1

    # Epilogue: scatter D.
    divisor = 1 if "limit_undivided" in faults else max(1, output_topk)
    limit = torch.tensor(Q32_RANGE, dtype=torch.float32) / float(divisor)
    for mt in range(t.mt):
        for i in range(8):
            r = m.d_row(mt, i, lanes)  # [32]
            orow = out_row[:, r]
            value = acc[:, :, :, mt, :, i] * weight[:, r][:, None, None, :]
            col = m.d_col(frag_n0[..., None], lanes)
            keep = (orow >= 0)[:, None, None, :] & (frag_n0[..., None] < size_n)
            sel_row = orow[:, None, None, :].expand_as(value)[keep]
            sel_pair = row_pair[:, r][:, None, None, :].expand_as(value)[keep]
            sel_col = col[keep]
            sel_val = value[keep]
            res.pair_value[sel_pair, sel_col] = sel_val
            res.writes.index_put_(
                (sel_pair, sel_col), torch.ones_like(sel_pair), accumulate=True
            )
            if not q32:
                res.out[sel_row, sel_col] = sel_val.to(dtype)
                continue
            bad = ~torch.isfinite(sel_val) | (sel_val.abs() >= limit)
            if "status_never_set" not in faults:
                res.out[sel_row[bad], size_n] = 1
            fixed = (sel_val[~bad].double() * Q32_SCALE).round().long()
            res.out.index_put_((sel_row[~bad], sel_col[~bad]), fixed, accumulate=True)
    return res


# ---------------------------------------------------------------------------
# Checks against the dense reference
# ---------------------------------------------------------------------------


def check_emulation(res: Emulation, case: Case, *, q32: bool) -> None:
    """Compare an emulation with the contract of the op, derived from the
    routing arrays and the packed weights alone."""
    dtype = case.dtype
    size_m = case.a.shape[0]
    num_experts, _, size_n = case.w.b_q.shape
    rows = scheduled_rows(case.routing, case.top_k, size_m, num_experts)
    w_rounded = rounded_weights(case.w, dtype)
    values, bound = reference_pair_values(case.a, w_rounded, rows, case.top_k)
    if case.topk_w is not None:
        wt = torch.stack([case.topk_w[p] for p, _ in rows]).double()[:, None]
        values, bound = values * wt, bound * wt.abs()
    pair_idx = torch.tensor([p for p, _ in rows], dtype=torch.int64)

    # Every scheduled row exactly once, nothing else touched.
    expected_writes = torch.zeros_like(res.writes)
    expected_writes[pair_idx] = 1
    assert torch.equal(res.writes, expected_writes), "row coverage differs"

    got = res.pair_value[pair_idx].double()
    finite = torch.isfinite(values)
    assert torch.equal(torch.isfinite(got), finite), "non-finite pattern differs"
    err = (got - values).abs()[finite]
    tol = 1e-5 * bound[finite] + 1e-12
    assert bool((err <= tol).all()), (
        f"pair values differ: max err {err.max().item():.3e} "
        f"(max allowed {tol.max().item():.3e})"
    )

    if not q32:
        out = res.out
        # One rounding of the fp32 value; untouched rows keep their poison.
        assert torch.equal(
            out[pair_idx].float().nan_to_num(nan=-1.0),
            res.pair_value[pair_idx].to(dtype).float().nan_to_num(nan=-1.0),
        )
        untouched = torch.ones(out.shape[0], dtype=torch.bool)
        untouched[pair_idx] = False
        assert bool(torch.isnan(out[untouched]).all()), (
            "a non-scheduled row was written"
        )
        return

    # Q32. The integer accumulator, element by element, from the model's own
    # pair values with the rule of the scalar kernel.
    limit = torch.tensor(Q32_RANGE, dtype=torch.float32) / max(1, case.output_topk)
    out_rows = pair_idx // case.output_topk if case.output_topk > 0 else pair_idx
    pv = res.pair_value[pair_idx]
    bad = ~torch.isfinite(pv) | (pv.abs() >= limit)
    safe = torch.where(bad, torch.zeros((), dtype=torch.float64), pv.double())
    expected = torch.zeros_like(res.out)
    expected[:, :size_n].index_add_(0, out_rows, (safe * Q32_SCALE).round().long())
    flagged = torch.zeros(case.out_rows, dtype=torch.int64)
    flagged.index_add_(0, out_rows, bad.any(dim=1).long())
    expected[:, size_n] = (flagged > 0).long()
    assert torch.equal(res.out, expected), "Q32 accumulator differs"

    # The sticky status from the dense reference (values far from the limit).
    ref_bad = ~finite | (values.abs() >= float(limit))
    ref_flagged = torch.zeros(case.out_rows, dtype=torch.int64)
    ref_flagged.index_add_(0, out_rows, ref_bad.any(dim=1).long())
    assert torch.equal(res.out[:, size_n], (ref_flagged > 0).long()), "row status"

    # The finished fp16 rows against the dense reference.
    ref = torch.zeros(case.out_rows, size_n, dtype=torch.float64)
    ref_bound = torch.zeros_like(ref)
    ref.index_add_(0, out_rows, values)
    ref_bound.index_add_(0, out_rows, bound)
    final = q32_finalize(res.out).double()
    scheduled_out = torch.zeros(case.out_rows, dtype=torch.bool)
    scheduled_out[out_rows] = True
    clean = scheduled_out & (res.out[:, size_n] == 0)
    got_rows, ref_rows = final[clean], ref[clean]
    diff = (got_rows - ref_rows).abs()
    allowed = (2.0**-9) * ref_rows.abs() + 2e-5 * ref_bound[clean] + 1e-6
    in_range = ref_rows.abs() < 6e4
    assert bool((diff[in_range] <= allowed[in_range]).all()), "finished Q32 rows"
    # Outside the fp16 range the finisher produces NaN (the band in between is
    # left alone: the reference is fp64, the accumulator is not).
    assert bool(torch.isnan(got_rows[ref_rows.abs() > 7e4]).all()), "overflow"
    assert bool(torch.isnan(final[res.out[:, size_n] != 0]).all()), "flagged: NaN"


def run_and_check(tile, mapping_cls, case, *, q32, faults=frozenset()):
    res = emulate_kernel(tile, mapping_cls, case, q32=q32, faults=faults)
    check_emulation(res, case, q32=q32)
    return res


# ---------------------------------------------------------------------------
# Tests: tile table, mapping, packing, dequant
# ---------------------------------------------------------------------------

DTYPES = [
    pytest.param(torch.float16, id="fp16"),
    pytest.param(torch.bfloat16, id="bf16"),
]
BM_PARAMS = [pytest.param(bm, id=f"bm{bm}") for bm in BLOCK_MS]


@pytest.mark.skipif(not KERNEL_SOURCE.exists(), reason="kernel source not in tree")
def test_tile_table_matches_the_kernel_source():
    assert parse_kernel_tiles(KERNEL_SOURCE.read_text()) == TILES


# Every index expression of ``Mapping`` (and the schedule facts the fault flags
# stand for) as it appears in the kernel source, whitespace aside.
INDEX_MATH_PINS = {
    "stage_row": [
        "const int stage_row = tid / VEC_PER_ROW;",
        "sorted_token_ids[row_base + stage_row + p * ROWS_PER_PASS]",
        "a_lds[buf][stage_row + p * ROWS_PER_PASS][stage_vec] = src[p];",
    ],
    "stage_vec": [
        "const int stage_vec = tid % VEC_PER_ROW;",
        "reinterpret_cast<const v4u32*>(a + (int64_t)token_row * size_k) + stage_vec",
        "a_src[p][k_base >> 3]",
    ],
    "frag_n0": [
        "const int tile_n0 = blockIdx.y * S::TILE_N;",
        "frag_n0[j] = tile_n0 + (wave * NF + j) * 16;",
        "if (frag_n0[j] >= size_n) continue;",
    ],
    "n_load": [
        "n_load[j] = min(frag_n0[j], size_n - 16) + lane_lo;",
        "const int lane_lo = lane & 15;",
        "const int lane_hi = lane >> 4;",
    ],
    "word_row": [
        "const int64_t b_row_bytes = (int64_t)size_n * 4;",
        "const uint32_t b_row_bytes32 = (uint32_t)size_n * 4u;",
        "b_col_byte[j] = (uint32_t)n_load[j] * 4u;",
        "const char* row = b_expert + (int64_t)(k >> 3) * b_row_bytes;",
        "row + (b_col_byte[j] + (uint32_t)r * b_row_bytes32)",
    ],
    "nibble": [
        "q0.u = ((qa >> 0) & 0x000F000F) | c0;",
        "q1.u = ((qa >> 4) & 0x000F000F) | c0;",
        "q2.u = ((qa >> 8) & 0x000F000F) | c0;",
        "q3.u = ((qa >> 12) & 0x000F000F) | c0;",
        "DQ::frag(cur[j][2 * ks], cur[j][2 * ks + 1], prep[j])",
    ],
    "zero_of": [
        "((zero_raw[j] >> ((n_load[j] & 7) * 4)) & 0xF) + 1",
        "zero_col_byte[j] = (uint32_t)(n_load[j] >> 3) * 4u;",
        "const char* zero_row = qz_expert + (int64_t)g * (size_n >> 3) * 4;",
    ],
    "scale_index": [
        "scale_col_byte[j] = (uint32_t)n_load[j] * (uint32_t)sizeof(T);",
        "sc_expert + (int64_t)g * size_n * (int64_t)sizeof(T)",
    ],
    "k_schedule": [
        "sub_step(k_base + 64 * pr, words_a, words_b, 8 * pr, buf);",
        "sub_step(k_base + 64 * pr + 32, words_b, words_a, 8 * pr + 4, buf);",
        "if (k >= size_k) return;",
        "load_words(nxt, min(k + 32, size_k - 32));",
        "for (int k_base = 0; k_base < size_k; k_base += KC)",
        "if (has_next) store_a_chunk(buf ^ 1, a_pref);",
    ],
    "group_refresh": [
        "int next_group_k = groupsize;",
        "if (k == next_group_k) {",
        "next_group_k += groupsize;",
        "load_group_raw(group + 1);",
        "g = min(g, groups - 1);",
    ],
    "a_fragment": [
        "a_lds[buf][16 * mt + lane_lo][kvec0 + 2 * ks]",
        "a_lds[buf][16 * mt + lane_lo][kvec0 + 2 * ks + 1]",
    ],
    "d_fragment": [
        "const int r = 16 * mt + 2 * i + lane_hi;",
        "const int n = frag_n0[j] + lane_lo;",
        "const float value = acc[mt][j][i] * weight;",
        "const float weight = topk_weights != nullptr ? s_weight[r] : 1.0f;",
    ],
    "routing": [
        "if (token_block * BLOCK_M >= num_tokens_post_padded[0]) return;",
        "if (expert_id < 0 || expert_id >= num_experts) return;",
        "const bool valid = pair >= 0 && pair / top_k < size_m;",
        "(pair >= 0 && token_row < size_m)",
        "s_out_row[tid] = valid ? (output_topk > 0 ? pair / output_topk : pair) : -1;",
        "(valid && topk_weights != nullptr) ? topk_weights[pair] : 0.0f;",
        "if (out_row < 0) continue;",
    ],
    "q32": [
        "1073741824.f / (float)max(1, output_topk)",
        "!isfinite(value) || fabsf(value) >= fixed_limit",
        "__float2ll_rn(value * 4294967296.f)",
        "long long* row = c + (int64_t)out_row * (size_n + 1);",
        "fixed_accum(value, row + n, row + size_n, fixed_limit);",
    ],
}


def _squash(text: str) -> str:
    return re.sub(r"\s+", "", text)


@pytest.mark.skipif(not KERNEL_SOURCE.exists(), reason="kernel source not in tree")
def test_kernel_source_still_contains_the_modelled_index_math():
    """An edit of an index expression of the kernel must come with the same
    edit of ``Mapping`` (and its mutants); this fails until both agree."""
    code = _squash(re.sub(r"//[^\n]*", "", KERNEL_SOURCE.read_text()))
    missing = {
        name: [pin for pin in pins if _squash(pin) not in code]
        for name, pins in INDEX_MATH_PINS.items()
    }
    missing = {name: pins for name, pins in missing.items() if pins}
    assert not missing, f"kernel source no longer contains: {missing}"


@pytest.mark.parametrize("bm", BM_PARAMS)
def test_tile_geometry_satisfies_the_kernel_static_asserts(bm):
    t = TILES[bm]
    assert bm % 16 == 0 and t.kc % 64 == 0
    assert t.threads % t.vec_per_row == 0 and bm % t.rows_per_pass == 0
    assert bm <= t.threads
    assert t.lds_bytes <= 32 * 1024
    assert t.passes * t.rows_per_pass == bm


@pytest.mark.parametrize("bm", BM_PARAMS)
def test_activation_staging_covers_the_chunk_exactly_once(bm):
    t = TILES[bm]
    m = Mapping(t)
    tids = torch.arange(t.threads)
    seen = torch.zeros(bm, t.vec_per_row, dtype=torch.int64)
    for p in range(t.passes):
        seen.index_put_(
            (m.stage_row(tids, p), m.stage_vec(tids)),
            torch.ones(t.threads, dtype=torch.int64),
            accumulate=True,
        )
    assert bool((seen == 1).all())


@pytest.mark.parametrize("bm", BM_PARAMS)
def test_fragment_columns_tile_the_n_range_exactly_once(bm):
    t = TILES[bm]
    m = Mapping(t)
    cols = torch.zeros(t.tile_n, dtype=torch.int64)
    lanes = torch.arange(LANES)
    for wave in range(t.waves):
        for j in range(t.nf):
            f0 = int(m.frag_n0(0, wave, j))
            # Both lane halves of a fragment name the same 16 columns.
            cols[m.d_col(f0, lanes)[:16]] += 1
            assert torch.equal(m.d_col(f0, lanes)[:16], m.d_col(f0, lanes)[16:])
    assert bool((cols == 1).all())


def test_wmma_d_layout_covers_each_accumulator_element_once():
    m = Mapping(TILES[16])
    lanes = torch.arange(LANES)
    seen = torch.zeros(16, 16, dtype=torch.int64)
    for i in range(8):
        seen[m.d_row(0, i, lanes), lanes & 15] += 1
    assert bool((seen == 1).all())


def test_shuffle_matches_the_documented_nibble_slots():
    w = make_weights(1, 64, 16, 32, torch.float16, torch.Generator().manual_seed(1))
    direct = torch.zeros(1, 8, 16, dtype=torch.int64)
    words = w.b_q.long() & 0xFFFFFFFF
    for s in range(8):  # slot s lives in nibble NIBBLE_OF_SLOT[s]
        direct |= (w.q[:, s::8, :] & 0xF) << (4 * NIBBLE_OF_SLOT[s])
    assert torch.equal(words, direct)
    assert [Mapping(TILES[16]).nibble(s) for s in range(8)] == NIBBLE_OF_SLOT


def test_packed_weights_decode_to_the_source_weights():
    gen = torch.Generator().manual_seed(2)
    w = make_weights(3, 96, 32, 32, torch.float16, gen)
    gs = w.group_size
    zeros = w.zeros.repeat_interleave(gs, dim=1) + 1
    scales = w.scales.double().repeat_interleave(gs, dim=1)
    assert torch.equal(dequantize_packed(w), (w.q - zeros).double() * scales)


@pytest.mark.parametrize("dtype", DTYPES)
def test_dequant_is_one_rounding_of_the_exact_product(dtype):
    """Every (nibble, zero) pair against scales of many binades."""
    gen = torch.Generator().manual_seed(3)
    nib = torch.arange(16)
    zero = torch.arange(1, 17)
    scale = (torch.rand(64, generator=gen) * 1.99 + 0.01).to(dtype)
    scale = scale * torch.tensor(2.0).pow(torch.randint(-6, 4, (64,), generator=gen))
    scale = scale.to(dtype)
    # words2 [Z, S, 2]: the 16 nibbles of a fragment in slot order.
    pos = torch.tensor(NIBBLE_OF_SLOT * 2)
    words = torch.zeros(1, 1, 2, dtype=torch.int64)
    for s in range(16):
        words[0, 0, s // 8] |= (s % 16) << (4 * NIBBLE_OF_SLOT[s % 8])
    for z in zero:
        got = dequant_b_fragment(
            words.expand(64, 1, 2),
            torch.full((64, 1), int(z)),
            scale[:, None],
            pos,
            dtype,
        )
        exact = (nib[None, :] - int(z)).double() * scale.double()[:, None]
        assert torch.equal(got[:, 0].double(), exact.to(dtype).double())


# ---------------------------------------------------------------------------
# Tests: the schedule against the dense reference
# ---------------------------------------------------------------------------

SHAPES = [
    # N tail (272 = 256 + 16), groups of 128
    pytest.param(dict(K=384, N=272, group_size=128), id="k384-n272-g128"),
    # K tail of the last chunk, a group boundary at every 32-K sub step
    pytest.param(dict(K=224, N=48, group_size=32), id="k224-n48-g32"),
    # one chunk, one group, one fragment
    pytest.param(dict(K=128, N=16, group_size=128), id="k128-n16-g128"),
    # down-projection K, five groups
    pytest.param(dict(K=640, N=96, group_size=128), id="k640-n96-g128"),
]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("bm", BM_PARAMS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("mul", [True, False], ids=["weighted", "plain"])
def test_direct_store_matches_the_dense_reference(shape, bm, dtype, mul):
    case = build_case(
        **shape, dtype=dtype, block_m=bm, M=11, top_k=3, E=4, mul=mul, seed=bm
    )
    run_and_check(TILES[bm], Mapping, case, q32=False)


@pytest.mark.parametrize("shape", SHAPES[:2])
@pytest.mark.parametrize("bm", BM_PARAMS)
@pytest.mark.parametrize("gemm2", [False, True], ids=["gate_up", "down"])
def test_q32_accumulator_matches_the_dense_reference(shape, bm, gemm2):
    case = build_case(
        **shape,
        dtype=torch.float16,
        block_m=bm,
        M=9,
        top_k=3,
        E=4,
        gemm2=gemm2,
        seed=7 + bm,
    )
    run_and_check(TILES[bm], Mapping, case, q32=True)


ROUTINGS = [
    pytest.param(dict(style="one_expert"), id="one_expert_many_blocks"),
    pytest.param(dict(style="sparse", E=6), id="empty_experts"),
    pytest.param(dict(drop=(1, 2)), id="expert_minus_one_blocks"),
    pytest.param(dict(stale_blocks=3), id="stale_blocks_past_ntpp"),
    pytest.param(dict(M=1), id="one_token"),
    pytest.param(dict(M=40, top_k=2, E=3, drop=(0,), stale_blocks=1), id="mixed"),
]


@pytest.mark.parametrize("routing", ROUTINGS)
@pytest.mark.parametrize("bm", BM_PARAMS)
@pytest.mark.parametrize("q32", [False, True], ids=["direct", "q32"])
def test_routing_edge_cases(routing, bm, q32):
    kwargs = dict(K=224, N=48, group_size=32, M=13, top_k=3, E=4, seed=11 + bm)
    kwargs.update(routing)
    case = build_case(dtype=torch.float16, block_m=bm, **kwargs)
    run_and_check(TILES[bm], Mapping, case, q32=q32)


@pytest.mark.parametrize("bm", BM_PARAMS)
@pytest.mark.parametrize("q32", [False, True], ids=["direct", "q32"])
def test_non_finite_activation_poisons_only_its_own_pairs(bm, q32):
    case = build_case(
        K=224, N=48, group_size=32, M=9, top_k=3, E=4, block_m=bm, nan_token=2, seed=5
    )
    res = run_and_check(TILES[bm], Mapping, case, q32=q32)
    poisoned = torch.isnan(res.pair_value).all(dim=1)
    assert poisoned.nonzero().flatten().tolist() == [6, 7, 8]  # token 2, top_k 3
    assert bool(torch.isfinite(res.pair_value[~poisoned]).all())
    if q32:
        assert res.out[:, -1].nonzero().flatten().tolist() == [6, 7, 8]


@pytest.mark.parametrize("bm", BM_PARAMS)
@pytest.mark.parametrize("gemm2", [False, True], ids=["gate_up", "down"])
def test_q32_limit_divides_by_output_topk_and_the_status_is_sticky(bm, gemm2):
    case = build_case(
        K=384,
        N=48,
        group_size=128,
        M=12,
        top_k=3,
        E=3,
        block_m=bm,
        gemm2=gemm2,
        hot_expert=1,
        seed=3,
    )
    res = run_and_check(TILES[bm], Mapping, case, q32=True)
    status = res.out[:, -1]
    assert 0 < int(status.sum()) < status.numel(), "scenario must mix both outcomes"


@pytest.mark.parametrize("dtype", DTYPES)
def test_direct_store_is_bitwise_independent_of_the_block_size(dtype):
    """One fp32 chain of K/16 WMMA steps per element, whatever the tile."""
    outs = {}
    for bm in BLOCK_MS:
        case = build_case(
            K=384, N=272, group_size=128, M=11, top_k=3, E=4, dtype=dtype, seed=9,
            block_m=bm,
        )  # fmt: skip
        outs[bm] = run_and_check(TILES[bm], Mapping, case, q32=False).out
    for bm in BLOCK_MS[1:]:
        assert torch.equal(
            outs[BLOCK_MS[0]].view(torch.int16), outs[bm].view(torch.int16)
        )


# ---------------------------------------------------------------------------
# Mutation tests: every deliberate index error must be caught
# ---------------------------------------------------------------------------


def _mutant(**overrides):
    return type("Mutant", (Mapping,), overrides)


MUTANTS = {
    # --- weights ---
    "nibble_slots_sequential": (_mutant(nibble=lambda self, s: s), frozenset()),
    "word_rows_swapped": (
        _mutant(word_row=lambda self, k, r: (k >> 3) + (r ^ 1)),
        frozenset(),
    ),
    "word_row_repeats_first_pair": (
        _mutant(word_row=lambda self, k, r: (k >> 3) + (r & 1)),
        frozenset(),
    ),
    "zero_offset_missing": (
        _mutant(zero_of=lambda self, zw, n: (zw >> ((n & 7) * 4)) & 0xF),
        frozenset(),
    ),
    "zero_nibble_by_n_mod_4": (
        _mutant(zero_of=lambda self, zw, n: ((zw >> ((n & 3) * 4)) & 0xF) + 1),
        frozenset(),
    ),
    "zero_word_index_n_div_16": (
        _mutant(zero_word_index=lambda self, g, n, N: g * (N >> 3) + (n >> 4)),
        frozenset(),
    ),
    "scale_index_ignores_group": (
        _mutant(scale_index=lambda self, g, n, N: n),
        frozenset(),
    ),
    "group_refresh_one_sub_step_late": (Mapping, frozenset({"group_late"})),
    # --- columns and the N tail ---
    "tail_fragment_not_clamped": (
        _mutant(n_load=lambda self, f, lane, N: f + (lane & 15)),
        frozenset(),
    ),
    "fragment_columns_overlap": (
        _mutant(
            frag_n0=lambda self, ny, w, j: (
                ny * self.t.tile_n + w * self.t.nf * 16 + j * 8
            )
        ),
        frozenset(),
    ),
    "d_col_uses_half_the_lanes": (
        _mutant(d_col=lambda self, f, lane: f + (lane >> 1)),
        frozenset(),
    ),
    # --- activation tile ---
    "stage_rows_alias": (
        _mutant(
            stage_row=lambda self, tid, p: (
                p * self.t.rows_per_pass + tid % self.t.rows_per_pass
            )
        ),
        frozenset(),
    ),
    # (A rotation of the vector index inside a row, e.g. (tid + 1) % vec_per_row,
    # is a bijection: the row is still staged completely, so it is no mutant.)
    "stage_vector_aliased": (
        _mutant(stage_vec=lambda self, tid: tid % (self.t.vec_per_row - 1)),
        frozenset(),
    ),
    "a_fragment_lane_half_not_masked": (
        _mutant(a_row=lambda self, mt, lane: 16 * mt + lane),
        frozenset(),
    ),
    "a_fragment_slot_halves_swapped": (
        _mutant(a_vec=lambda self, v, ks, half: v + 2 * ks + (1 - half)),
        frozenset(),
    ),
    "a_fragment_ignores_k_step": (
        _mutant(a_vec=lambda self, v, ks, half: v + half),
        frozenset(),
    ),
    "second_sub_step_vector_offset": (
        _mutant(kvec0=lambda self, pr, second: 8 * pr + (3 if second else 0)),
        frozenset(),
    ),
    "second_sub_step_k_offset": (
        _mutant(
            sub_k=lambda self, kb, pr, second: kb + 64 * pr + (16 if second else 0)
        ),
        frozenset(),
    ),
    "lds_store_to_the_live_buffer": (Mapping, frozenset({"store_same_buffer"})),
    # --- output ---
    "d_rows_blocked_not_interleaved": (
        _mutant(d_row=lambda self, mt, i, lane: 16 * mt + i + 8 * (lane >> 4)),
        frozenset(),
    ),
    "d_row_ignores_lane_half": (
        _mutant(d_row=lambda self, mt, i, lane: 16 * mt + 2 * i + 0 * lane),
        frozenset(),
    ),
    "out_row_ignores_output_topk": (
        _mutant(out_row=lambda self, pair, output_topk: pair),
        frozenset(),
    ),
    "topk_weight_read_by_tile_row": (
        _mutant(weight_index=lambda self, pair, r: r),
        frozenset(),
    ),
    "topk_weight_dropped": (Mapping, frozenset({"weight_dropped"})),
    # --- routing ---
    "num_tokens_post_padded_ignored": (Mapping, frozenset({"ignore_ntpp"})),
    "expert_minus_one_not_skipped": (Mapping, frozenset({"no_expert_check"})),
    "padding_rows_not_checked_in_staging": (
        Mapping,
        frozenset({"no_valid_check_stage"}),
    ),
    "padding_rows_not_checked_in_epilogue": (
        Mapping,
        frozenset({"no_valid_check_meta"}),
    ),
    # --- Q32 ---
    "q32_limit_not_divided_by_output_topk": (Mapping, frozenset({"limit_undivided"})),
    "q32_status_never_set": (Mapping, frozenset({"status_never_set"})),
}

# Scenarios that together exercise every feature the mutants break.
MUTATION_SCENARIOS = [
    (
        dict(
            K=384,
            N=272,
            group_size=128,
            M=13,
            top_k=3,
            E=4,
            mul=True,
            drop=(2,),
            stale_blocks=2,
            seed=21,
        ),
        False,
    ),  # fmt: skip
    (dict(K=224, N=48, group_size=32, M=9, top_k=2, E=3, mul=True, seed=22), False),
    (
        dict(
            K=384,
            N=48,
            group_size=128,
            M=12,
            top_k=3,
            E=3,
            gemm2=True,
            hot_expert=1,
            seed=23,
        ),
        True,
    ),  # fmt: skip
]
CAUGHT = (AssertionError, IndexError, RuntimeError)


@pytest.mark.parametrize("bm", BM_PARAMS)
def test_mutation_scenarios_pass_unmutated(bm):
    for kwargs, q32 in MUTATION_SCENARIOS:
        case = build_case(dtype=torch.float16, block_m=bm, **kwargs)
        run_and_check(TILES[bm], Mapping, case, q32=q32)


@pytest.mark.parametrize("bm", BM_PARAMS)
@pytest.mark.parametrize("name", list(MUTANTS))
def test_mutants_are_caught(name, bm):
    mapping_cls, faults = MUTANTS[name]
    for kwargs, q32 in MUTATION_SCENARIOS:
        case = build_case(dtype=torch.float16, block_m=bm, **kwargs)
        try:
            run_and_check(TILES[bm], mapping_cls, case, q32=q32, faults=faults)
        except CAUGHT:
            return
    pytest.fail(f"mutant {name!r} survived at block_size_m {bm}")
