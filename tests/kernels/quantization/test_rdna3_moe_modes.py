# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the fused W4A16 MoE modes on gfx1100.

No GPU is used. The HIP ops are replaced by pure-torch stand-ins that follow
the kernels' accumulation contract, so these tests pin the Q32 fixed-point
contract, the hub masking contract and the call flow of
``Rdna3WNA16Experts.apply``. The kernel arithmetic itself is checked against
the same references by the gfx1100-gated tests in ``test_rdna3_moe_w4a16.py``.
"""

import copy
import functools

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch._subclasses.fake_tensor import (
    DataDependentOutputException,
    DynamicOutputShapeException,
    FakeTensorMode,
)

import vllm._custom_ops as ops
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
    int4_w4a16_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.experts import rdna3_moe
from vllm.model_executor.layers.fused_moe.experts.rdna3_moe import Rdna3WNA16Experts

Q32 = float(1 << 32)
# K columns handled by one kernel block (BLOCK_KN_SIZE in moe_q_gemm_rdna3.cu).
K_SLICE = 256
HIDDEN = 320
INTER = 288
TOP_K = 10


# --- Q32 reference: mirrors fixed_accum_row and the cast kernel ------------


def q32_limit(k_slices: int, output_topk: int) -> torch.Tensor:
    """Largest accepted |partial|: 2^30 / (K slices * max(1, output_topk))."""
    return torch.tensor(2.0**30) / torch.tensor(float(k_slices * max(1, output_topk)))


def q32_add(accumulator: torch.Tensor, partial: torch.Tensor, limit: torch.Tensor):
    """One kernel block's update of ``[rows, N + 1]`` with fp32 ``[rows, N]``.

    A non-finite or out-of-bound partial sets the row's status instead of
    being added; the rest of the row is still accumulated.
    """
    bad = ~torch.isfinite(partial) | (partial.abs() >= limit)
    accumulator[:, -1] |= bad.any(dim=1).to(torch.int64)
    kept = partial.masked_fill(bad, 0.0)
    accumulator[:, :-1] += torch.round(kept * Q32).to(torch.int64)


def q32_finalize(accumulator: torch.Tensor) -> torch.Tensor:
    """int64 -> fp32 -> fp16; NaN for a flagged row or an fp16 overflow."""
    value = (accumulator[:, :-1].to(torch.float32) * (1.0 / Q32)).to(torch.float16)
    poisoned = (accumulator[:, -1:] != 0) | ~torch.isfinite(value)
    return value.masked_fill(poisoned, float("nan"))


def q32_sum(partials: torch.Tensor, order: torch.Tensor, limit: torch.Tensor):
    """Accumulate ``partials[i]`` (``[rows, N]``) in the given order."""
    rows, columns = partials.shape[1:]
    accumulator = torch.zeros(rows, columns + 1, dtype=torch.int64)
    for index in order.tolist():
        q32_add(accumulator, partials[index], limit)
    return accumulator


def fp16_cas_sum(partials: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
    """Sequential fp16 adds: the arrival-order accumulation of the default mode."""
    total = torch.zeros(partials.shape[1:], dtype=torch.float16)
    for index in order.tolist():
        total = total + partials[index].to(torch.float16)
    return total


def float64_to_fp16(value: torch.Tensor) -> torch.Tensor:
    """Round a float64 tensor to fp16 once (torch would round via fp32)."""
    return torch.from_numpy(value.numpy().astype(np.float16))


def random_partials(seed: int, count: int, rows: int, columns: int) -> torch.Tensor:
    """fp32 partials that span several binades per column."""
    generator = torch.Generator().manual_seed(seed)
    exponent = torch.randint(-8, 4, (1, 1, columns), generator=generator)
    return torch.randn(count, rows, columns, generator=generator) * torch.exp2(
        exponent.float()
    )


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("slices,top_k", [(1, 1), (3, 10), (10, 1), (10, 10)])
def test_q32_sum_is_independent_of_partial_order(seed, slices, top_k):
    count = slices * top_k
    partials = random_partials(seed, count, 4, 96)
    limit = q32_limit(slices, top_k)
    reference = q32_sum(partials, torch.arange(count), limit)
    generator = torch.Generator().manual_seed(100 + seed)
    for _ in range(5):
        order = torch.randperm(count, generator=generator)
        shuffled = q32_sum(partials, order, limit)
        assert torch.equal(reference, shuffled)
        assert torch.equal(q32_finalize(reference), q32_finalize(shuffled))


def test_q32_equals_float64_reference_rounded_to_fp16_when_sums_are_fp32_exact():
    generator = torch.Generator().manual_seed(0)
    # |sum| < 30 * 2^18 < 2^23 units of 2^-32: int64 -> fp32 is exact, so the
    # finalize rounds the exact float64 value once, like numpy does.
    units = torch.randint(-(1 << 18) + 1, 1 << 18, (30, 6, 128), generator=generator)
    partials = units.to(torch.float32) / Q32
    accumulator = q32_sum(partials, torch.arange(30), q32_limit(3, 10))
    assert torch.equal(accumulator[:, :-1], units.sum(0))
    assert not accumulator[:, -1].any()
    expected = float64_to_fp16(units.sum(0).double() / Q32)
    assert torch.equal(q32_finalize(accumulator), expected)


def test_q32_finalize_rounds_through_fp32_like_the_rdna2_cast():
    one_plus = (1 << 32) + (1 << 21) + 1  # 1 + 2^-11 + 2^-32
    accumulator = torch.tensor([[one_plus, -one_plus, 0]], dtype=torch.int64)
    # int64 -> fp32 drops the 2^-32 and lands exactly on the fp16 tie, which
    # rounds to even ...
    assert q32_finalize(accumulator).tolist() == [[1.0, -1.0]]
    # ... while rounding the exact value once goes to the next fp16 number.
    exact = float64_to_fp16(accumulator[:, :-1].double() / Q32)
    assert exact.tolist() == [[1.0009765625, -1.0009765625]]


@pytest.mark.parametrize("seed", range(3))
def test_q32_finalize_is_float64_rounding_except_at_double_rounding_ties(seed):
    partials = random_partials(seed, 30, 16, 512)
    accumulator = q32_sum(partials, torch.arange(30), q32_limit(3, 10))
    got = q32_finalize(accumulator)
    exact = accumulator[:, :-1].double() / Q32
    direct = float64_to_fp16(exact)
    differing = got != direct
    # A mismatch only happens when the fp32 intermediate sits exactly on the
    # midpoint of the two neighbouring fp16 numbers.
    fp32 = accumulator[:, :-1].to(torch.float32).double() / Q32
    midpoint = (got.double() + direct.double()) / 2
    assert torch.equal(fp32[differing], midpoint[differing])
    assert differing.float().mean() < 1e-3


def test_q32_error_against_the_float64_sum_of_the_partials_is_bounded():
    count = 30
    partials = random_partials(7, count, 16, 512)
    accumulator = q32_sum(partials, torch.arange(count), q32_limit(3, 10))
    got = q32_finalize(accumulator).double()
    exact = partials.double().sum(0)
    # fp16 half-ulp + fp32 conversion + 2^-33 quantization per partial.
    bound = exact.abs() * (2.0**-11 + 2.0**-23) + count * 2.0**-32 + 2.0**-24
    assert (got - exact).abs().le(bound).all()


@pytest.mark.parametrize("bad", ["nan", "inf", "neg_inf", "limit", "neg_limit"])
def test_q32_bad_partial_poisons_only_its_row_and_stays_sticky(bad):
    limit = q32_limit(3, 10)
    value = {
        "nan": float("nan"),
        "inf": float("inf"),
        "neg_inf": -float("inf"),
        "limit": float(limit),
        "neg_limit": -float(limit),
    }[bad]
    accumulator = torch.zeros(3, 5, dtype=torch.int64)
    partial = torch.full((3, 4), 0.25)
    partial[0, 2] = value
    q32_add(accumulator, partial, limit)
    assert accumulator[:, -1].tolist() == [1, 0, 0]
    # A later valid partial neither clears the flag nor touches other rows.
    q32_add(accumulator, torch.full((3, 4), 0.25), limit)
    assert accumulator[:, -1].tolist() == [1, 0, 0]
    result = q32_finalize(accumulator)
    assert result[0].isnan().all()
    assert result[1:].tolist() == [[0.5] * 4] * 2


def test_q32_fp16_overflow_becomes_nan():
    accumulator = torch.zeros(1, 4, dtype=torch.int64)
    accumulator[0, :3] = torch.tensor([65504, 65520, -65520]) * (1 << 32)
    assert q32_finalize(accumulator)[0].isnan().tolist() == [False, True, True]


@pytest.mark.parametrize("slices,top_k", [(1, 1), (10, 1), (3, 10), (10, 10)])
def test_q32_limit_keeps_the_sum_inside_int64(slices, top_k):
    limit = q32_limit(slices, top_k)
    just_below = torch.nextafter(limit, torch.zeros(()))
    accumulator = torch.zeros(1, 2, dtype=torch.int64)
    for _ in range(slices * max(1, top_k)):
        q32_add(accumulator, just_below.reshape(1, 1), limit)
    assert accumulator[0, -1] == 0
    assert 2**61 < int(accumulator[0, 0]) < 2**63


def test_fp16_arrival_order_sums_differ_while_q32_does_not():
    partials = torch.tensor([2048.0, 1.0, 1.0]).reshape(3, 1, 1)
    big_first = fp16_cas_sum(partials, torch.tensor([0, 1, 2]))
    small_first = fp16_cas_sum(partials, torch.tensor([1, 2, 0]))
    assert (big_first.item(), small_first.item()) == (2048.0, 2050.0)
    limit = q32_limit(1, 1)
    for order in ([0, 1, 2], [1, 2, 0], [2, 1, 0]):
        accumulator = q32_sum(partials, torch.tensor(order), limit)
        assert q32_finalize(accumulator).item() == 2050.0


# --- Hub-contract MoE reference -------------------------------------------


@functools.cache
def expert_w1(expert: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(1_000 + expert)
    return torch.randn(HIDDEN, 2 * INTER, generator=generator) / HIDDEN**0.5


@functools.cache
def expert_w2(expert: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(5_000 + expert)
    return torch.randn(INTER, HIDDEN, generator=generator) / INTER**0.5


def make_slot_map(global_experts: int, residents: list[int]) -> torch.Tensor:
    """Hub ``slot_map``: global expert id -> local slot, -1 if not resident."""
    slot_map = torch.full((global_experts,), -1, dtype=torch.int32)
    slot_map[torch.tensor(residents)] = torch.arange(len(residents), dtype=torch.int32)
    return slot_map


def hub_reference(x, topk_ids, topk_weights, slot_map, w1_slots, w2_slots, on_input):
    """float64 MoE over the resident experts; ``-1`` ids and absent experts skip."""
    out = torch.zeros(x.shape, dtype=torch.float64)
    for token in range(x.size(0)):
        for k in range(topk_ids.size(1)):
            expert = int(topk_ids[token, k])
            slot = int(slot_map[expert]) if expert >= 0 else -1
            if slot < 0:
                continue
            weight = float(topk_weights[token, k])
            hidden = x[token].double() * (weight if on_input else 1.0)
            gate, up = (hidden @ w1_slots[slot].double()).chunk(2)
            expert_out = (F.silu(gate) * up) @ w2_slots[slot].double()
            out[token] += expert_out if on_input else weight * expert_out
    return out


class Case:
    """Routing inputs of one MoE call and the experts resident on this rank."""

    def __init__(
        self, seed, tokens, *, global_experts=24, pool=24, masked=0.0, residents=None
    ):
        generator = torch.Generator().manual_seed(seed)
        self.x = torch.randn(tokens, HIDDEN, generator=generator).to(torch.float16)
        ids = [torch.randperm(pool, generator=generator)[:TOP_K] for _ in range(tokens)]
        self.ids = torch.stack(ids).to(torch.int32)
        self.ids[torch.rand(self.ids.shape, generator=generator) < masked] = -1
        self.weights = torch.softmax(
            torch.randn(tokens, TOP_K, generator=generator), dim=-1
        )
        self.global_experts = global_experts
        self.residents = list(range(pool)) if residents is None else residents
        self.slot_map = make_slot_map(global_experts, self.residents)
        self.w1 = torch.stack([expert_w1(e) for e in self.residents])
        self.w2 = torch.stack([expert_w2(e) for e in self.residents])

    def restricted(self, residents: list[int]) -> "Case":
        """The same routing on a rank that only holds ``residents``."""
        other = copy.copy(self)
        other.residents = residents
        other.slot_map = make_slot_map(self.global_experts, residents)
        other.w1 = torch.stack([expert_w1(e) for e in residents])
        other.w2 = torch.stack([expert_w2(e) for e in residents])
        return other

    def reference(self, on_input: bool = False) -> torch.Tensor:
        return hub_reference(
            self.x,
            self.ids,
            self.weights,
            self.slot_map,
            self.w1,
            self.w2,
            on_input,
        )


def relative_l2(got: torch.Tensor, want: torch.Tensor) -> float:
    return float((got.double() - want).norm() / want.norm())


def test_hub_reference_ignores_masked_and_non_resident_pairs():
    case = Case(0, 6, masked=0.3, residents=[1, 4, 5, 9, 12, 13, 20])
    keep = (case.ids >= 0) & (case.slot_map[case.ids.clamp(min=0).long()] >= 0)
    assert 0 < keep.sum() < keep.numel()
    everyone = case.restricted(list(range(24)))
    expected = hub_reference(
        case.x,
        case.ids.clamp(min=0),
        case.weights * keep,
        everyone.slot_map,
        everyone.w1,
        everyone.w2,
        False,
    )
    torch.testing.assert_close(case.reference(), expected, rtol=1e-12, atol=1e-12)


def test_partial_sum_identity_of_resident_subset_and_complement():
    case = Case(1, 8, masked=0.1)
    subset = [0, 2, 3, 7, 11, 12, 18, 19]
    complement = [e for e in range(24) if e not in subset]
    full = case.reference()
    part_a = case.restricted(subset).reference()
    part_b = case.restricted(complement).reference()
    torch.testing.assert_close(part_a + part_b, full, rtol=1e-12, atol=1e-12)
    a, b, f = part_a.half(), part_b.half(), full.half()
    bound = 2.0**-10 * (a.abs() + b.abs() + f.abs()).float() + 1e-6
    assert ((a + b).float() - f.float()).abs().le(bound).all()


# --- Pure-torch stand-ins for the gfx1100 ops ------------------------------


class FakeHip:
    """Replaces the three native ops, follows their contract, records calls."""

    def __init__(self, order_seed=None):
        self.events = []
        self.dense = {}
        self.order_seed = order_seed
        self.align_map = None
        self.accumulators = []
        self.pairs_seen = []

    def register(self, packed: torch.Tensor, dense: torch.Tensor) -> None:
        self.dense[packed.data_ptr()] = dense

    def install(self, monkeypatch) -> None:
        rocm = torch.ops._rocm_C
        monkeypatch.setattr(rocm, "moe_gptq_gemm_rdna3", self.gemm, raising=False)
        monkeypatch.setattr(
            rocm, "moe_fixedpoint_to_half_rdna3", self.cast, raising=False
        )
        monkeypatch.setattr(
            rocm, "moe_gptq_gemm_rdna3_wmma", self.wmma_gemm, raising=False
        )
        monkeypatch.setattr(ops, "moe_align_block_size", self.align)

    def align(
        self,
        topk_ids,
        num_experts,
        block_size,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
        expert_map=None,
        scatter_idx=None,
    ):
        """moe_align_block_size_kernel: ids outside [0, E) are skipped *before*
        the map is applied, and a map entry of -1 drops the pair too."""
        self.events.append(
            {
                "op": "align",
                "num_experts": num_experts,
                "block_size": block_size,
                "expert_map": expert_map is not None,
            }
        )
        self.align_map = expert_map
        flat = topk_ids.reshape(-1).to(torch.int64)
        local = torch.where((flat < 0) | (flat >= num_experts), -1, flat)
        if expert_map is not None:
            mapped = expert_map[local.clamp(min=0)].to(torch.int64)
            local = torch.where(local >= 0, mapped, -1)
        assert bool((local < num_experts).all())
        sorted_ids.fill_(flat.numel())
        expert_ids.fill_(-1)
        position = block = 0
        for expert in range(num_experts):
            pairs = (local == expert).nonzero().flatten()
            if pairs.numel() == 0:
                continue
            blocks = -(-pairs.numel() // block_size)
            sorted_ids[position : position + pairs.numel()] = pairs.to(torch.int32)
            expert_ids[block : block + blocks] = expert
            position += blocks * block_size
            block += blocks
        num_tokens_post_pad[0] = position

    def gemm(
        self,
        a,
        c,
        b_q_weight,
        b_scales,
        b_qzeros,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        top_k,
        block_size_m,
        mul_topk_weight,
        output_topk,
    ):
        """moe_gptq_gemm_rdna3: host checks, then one partial per (pair, K slice)
        accumulated into row ``pair // output_topk`` (or ``pair``)."""
        fixed = c.dtype == torch.int64
        size_n = b_q_weight.size(2)
        pairs = a.size(0) * top_k
        if fixed:
            assert a.dtype == torch.float16
            rows = -(-pairs // output_topk) if output_topk > 0 else pairs
            assert c.is_contiguous() and tuple(c.shape) == (rows, size_n + 1)
        else:
            assert c.dtype == a.dtype and c.size(1) == size_n
        assert torch.count_nonzero(c) == 0, "accumulator must be zero on entry"
        self.events.append(
            {
                "op": "gemm",
                "fixed": fixed,
                "a_shape": tuple(a.shape),
                "c_shape": tuple(c.shape),
                "top_k": top_k,
                "block_size_m": block_size_m,
                "mul_topk_weight": mul_topk_weight,
                "output_topk": output_topk,
                "topk_weights": topk_weights.numel(),
            }
        )
        slices = -(-a.size(1) // K_SLICE)
        limit = q32_limit(slices, output_topk)
        dense = self.dense[b_q_weight.data_ptr()]
        work: list[tuple[int, int, int]] = []
        for block in range(sorted_token_ids.numel() // block_size_m):
            if block * block_size_m >= int(num_tokens_post_padded[0]):
                continue
            expert = int(expert_ids[block])
            if expert < 0:
                continue
            for lane in range(block_size_m):
                pair = int(sorted_token_ids[block * block_size_m + lane])
                if pair < pairs:
                    work.extend((pair, expert, piece) for piece in range(slices))
        self.pairs_seen.append(sorted({pair for pair, _, _ in work}))
        if self.order_seed is not None:
            generator = torch.Generator().manual_seed(
                self.order_seed + len(self.events)
            )
            order = torch.randperm(len(work), generator=generator).tolist()
            work = [work[i] for i in order]
        for pair, expert, piece in work:
            low, high = piece * K_SLICE, min((piece + 1) * K_SLICE, a.size(1))
            row_a = a[pair // top_k, low:high].double()
            partial = (row_a.unsqueeze(1) * dense[expert, low:high].double()).sum(0)
            partial = partial.float()
            if mul_topk_weight:
                partial = partial * topk_weights[pair]
            row = pair // output_topk if output_topk > 0 else pair
            if fixed:
                q32_add(c[row : row + 1], partial.unsqueeze(0), limit)
            else:
                c[row] = c[row] + partial.to(c.dtype)

    def wmma_gemm(
        self,
        a,
        c,
        b_q_weight,
        b_scales,
        b_qzeros,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        top_k,
        block_size_m,
        mul_topk_weight,
        output_topk=0,
    ):
        """moe_gptq_gemm_rdna3_wmma: one full-K result per scheduled pair. With
        ``output_topk == 0`` it is stored into row ``pair`` (nothing is read or
        zeroed, unscheduled rows keep their bytes); otherwise it is added into
        row ``pair // output_topk`` of the zeroed int64 Q32 scratch (fp16 only).
        """
        assert block_size_m in (16, 32, 64)
        fixed = output_topk > 0
        size_n = b_q_weight.size(2)
        pairs = a.size(0) * top_k
        if fixed:
            assert a.dtype == torch.float16 and c.dtype == torch.int64
            rows = -(-pairs // output_topk)
            assert c.is_contiguous() and tuple(c.shape) == (rows, size_n + 1)
            assert torch.count_nonzero(c) == 0, "accumulator must be zero on entry"
        else:
            assert c.dtype == a.dtype and tuple(c.shape) == (pairs, size_n)
        self.events.append(
            {
                "op": "wmma_gemm",
                "fixed": fixed,
                "a_shape": tuple(a.shape),
                "c_shape": tuple(c.shape),
                "top_k": top_k,
                "block_size_m": block_size_m,
                "mul_topk_weight": mul_topk_weight,
                "output_topk": output_topk,
                "topk_weights": topk_weights.numel(),
            }
        )
        limit = q32_limit(1, output_topk)
        dense = self.dense[b_q_weight.data_ptr()]
        work: list[tuple[int, int]] = []
        for block in range(sorted_token_ids.numel() // block_size_m):
            if block * block_size_m >= int(num_tokens_post_padded[0]):
                continue
            expert = int(expert_ids[block])
            if expert < 0:
                continue
            for lane in range(block_size_m):
                pair = int(sorted_token_ids[block * block_size_m + lane])
                if pair < pairs:
                    work.append((pair, expert))
        self.pairs_seen.append(sorted(pair for pair, _ in work))
        if self.order_seed is not None:
            generator = torch.Generator().manual_seed(
                self.order_seed + len(self.events)
            )
            order = torch.randperm(len(work), generator=generator).tolist()
            work = [work[i] for i in order]
        for pair, expert in work:
            row_a = a[pair // top_k].double()
            partial = (row_a.unsqueeze(1) * dense[expert].double()).sum(0).float()
            if mul_topk_weight:
                partial = partial * topk_weights[pair]
            if fixed:
                row = pair // output_topk
                q32_add(c[row : row + 1], partial.unsqueeze(0), limit)
            else:
                c[pair] = partial.to(c.dtype)

    def cast(self, accumulator, output):
        """moe_fixedpoint_to_half_rdna3: writes every element of ``output``."""
        assert accumulator.dtype == torch.int64 and output.dtype == torch.float16
        assert accumulator.is_contiguous() and output.is_contiguous()
        assert accumulator.size(0) == output.size(0)
        assert accumulator.size(1) == output.size(1) + 1
        self.events.append(
            {
                "op": "cast",
                "accumulator_shape": tuple(accumulator.shape),
                "output_shape": tuple(output.shape),
            }
        )
        self.accumulators.append(accumulator.clone())
        output.copy_(q32_finalize(accumulator))

    def activation(self, activation, output, input, **kwargs):
        self.events.append(
            {
                "op": "activation",
                "input_shape": tuple(input.shape),
                "output_shape": tuple(output.shape),
            }
        )
        gate, up = input.float().chunk(2, dim=-1)
        output.copy_((F.silu(gate) * up).to(output.dtype))


def make_experts(
    monkeypatch,
    case,
    *,
    deterministic=False,
    block_m=None,
    wmma=False,
    wmma_min_m=None,
    wmma_block_m=None,
    wmma_op=True,
):
    """A real ``Rdna3WNA16Experts``; the env knobs are read by its constructor.

    ``wmma_op`` says whether the WMMA op counts as built into the binary.
    """
    if deterministic:
        monkeypatch.setenv("VLLM_RDNA3_MOE_DETERMINISTIC", "1")
    else:
        monkeypatch.delenv("VLLM_RDNA3_MOE_DETERMINISTIC", raising=False)
    if block_m is None:
        monkeypatch.delenv("VLLM_RDNA3_MOE_BLOCK_M", raising=False)
    else:
        monkeypatch.setenv("VLLM_RDNA3_MOE_BLOCK_M", str(block_m))
    for name, value in (
        ("VLLM_RDNA3_MOE_WMMA", "1" if wmma else None),
        ("VLLM_RDNA3_MOE_WMMA_MIN_M", wmma_min_m),
        ("VLLM_RDNA3_MOE_WMMA_BLOCK_M", wmma_block_m),
    ):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, str(value))
    monkeypatch.setattr(rdna3_moe, "rdna3_moe_wmma_available", lambda: wmma_op)
    moe_config = FusedMoEConfig(
        num_experts=case.global_experts,
        experts_per_token=TOP_K,
        hidden_dim=HIDDEN,
        intermediate_size=INTER,
        num_local_experts=len(case.residents),
        num_logical_experts=case.global_experts,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        activation=MoEActivation.SILU,
        in_dtype=torch.float16,
        device="cpu",
        routing_method=RoutingMethodType.TopK,
        max_num_tokens=2048,
    )
    scales = torch.ones(1, dtype=torch.float16)
    zeros = torch.zeros(1, dtype=torch.int32)
    quant_config = int4_w4a16_moe_quant_config(
        w1_scale=scales, w2_scale=scales, w1_zp=zeros, w2_zp=zeros
    )
    return Rdna3WNA16Experts(moe_config, quant_config)


def apply_case(
    monkeypatch,
    case,
    *,
    deterministic=False,
    block_m=None,
    order_seed=None,
    on_input=False,
    use_map=True,
    dtype=torch.float16,
    wmma=False,
    wmma_min_m=None,
    wmma_block_m=None,
):
    """Run ``apply`` on garbage-filled buffers; returns (output, FakeHip)."""
    hip = FakeHip(order_seed)
    hip.install(monkeypatch)
    experts = make_experts(
        monkeypatch,
        case,
        deterministic=deterministic,
        block_m=block_m,
        wmma=wmma,
        wmma_min_m=wmma_min_m,
        wmma_block_m=wmma_block_m,
    )
    experts.activation = hip.activation
    local, tokens = len(case.residents), case.x.size(0)
    w1 = torch.zeros(local, HIDDEN // 8, 2 * INTER, dtype=torch.int32)
    w2 = torch.zeros(local, INTER // 8, HIDDEN, dtype=torch.int32)
    hip.register(w1, case.w1)
    hip.register(w2, case.w2)
    output = torch.full((tokens, HIDDEN), 7.0, dtype=dtype)
    experts.apply(
        output,
        case.x.to(dtype),
        w1,
        w2,
        case.weights,
        case.ids,
        MoEActivation.SILU,
        case.global_experts,
        case.slot_map if use_map else None,
        None,
        None,
        torch.empty(tokens, HIDDEN, dtype=dtype),
        torch.full((tokens * TOP_K * 3 * INTER,), 3.0, dtype=dtype),
        None,
        on_input,
    )
    return output, hip


# --- Rdna3WNA16Experts.apply on the stand-ins ------------------------------


def expected_flow(case, tokens, *, fixed, block_size, on_input=False):
    pairs = tokens * TOP_K
    gate_up = 2 * INTER
    extra = 1 if fixed else 0
    align = {
        "op": "align",
        "num_experts": case.global_experts,
        "block_size": block_size,
        "expert_map": True,
    }

    def gemm(a_shape, c_shape, top_k, mul, output_topk, weights):
        return {
            "op": "gemm",
            "fixed": fixed,
            "a_shape": a_shape,
            "c_shape": c_shape,
            "top_k": top_k,
            "block_size_m": block_size,
            "mul_topk_weight": mul,
            "output_topk": output_topk,
            "topk_weights": weights,
        }

    def cast(rows, columns):
        return {
            "op": "cast",
            "accumulator_shape": (rows, columns + 1),
            "output_shape": (rows, columns),
        }

    activation = {
        "op": "activation",
        "input_shape": (pairs, gate_up),
        "output_shape": (pairs, INTER),
    }
    first = gemm(
        (tokens, HIDDEN),
        (pairs, gate_up + extra),
        TOP_K,
        on_input,
        0,
        pairs if on_input else 0,
    )
    second = gemm(
        (pairs, INTER),
        (tokens, HIDDEN + extra),
        1,
        not on_input,
        TOP_K,
        0 if on_input else pairs,
    )
    if not fixed:
        return [align, first, activation, second]
    return [
        align,
        first,
        cast(pairs, gate_up),
        activation,
        second,
        cast(tokens, HIDDEN),
    ]


@pytest.mark.parametrize("on_input", [False, True])
@pytest.mark.parametrize("tokens,block_size", [(1, 1), (3, 1), (4, 1), (5, 4), (24, 4)])
def test_default_mode_flow_zeroes_destinations_and_keeps_argument_order(
    monkeypatch, tokens, block_size, on_input
):
    case = Case(2, tokens, global_experts=512, pool=24, masked=0.1)
    output, hip = apply_case(monkeypatch, case, on_input=on_input)
    assert hip.events == expected_flow(
        case, tokens, fixed=False, block_size=block_size, on_input=on_input
    )
    assert relative_l2(output, case.reference(on_input)) < 5e-3


@pytest.mark.parametrize("on_input", [False, True])
@pytest.mark.parametrize("tokens,block_size", [(1, 1), (3, 1), (4, 1), (5, 4), (24, 4)])
def test_q32_mode_flow_uses_int64_scratch_and_casts_every_output(
    monkeypatch, tokens, block_size, on_input
):
    case = Case(2, tokens, global_experts=512, pool=24, masked=0.1)
    output, hip = apply_case(monkeypatch, case, deterministic=True, on_input=on_input)
    assert hip.events == expected_flow(
        case, tokens, fixed=True, block_size=block_size, on_input=on_input
    )
    assert relative_l2(output, case.reference(on_input)) < 5e-3
    assert not bool((output == 7.0).any()), "the cast must overwrite the output"


@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize("tokens", [1, 3, 24])
def test_hub_contract_masked_ids_and_slot_map(monkeypatch, tokens, deterministic):
    residents = [0, 3, 4, 8, 9, 15, 16, 22, 30, 33, 36, 39]
    case = Case(
        3, tokens, global_experts=512, pool=40, masked=0.25, residents=residents
    )
    keep = (case.ids >= 0) & (case.slot_map[case.ids.clamp(min=0).long()] >= 0)
    assert 0 < keep.sum() < keep.numel()
    output, hip = apply_case(monkeypatch, case, deterministic=deterministic)
    assert relative_l2(output, case.reference()) < 5e-3
    # The map reaches the align op untouched, and only resident, unmasked
    # pairs are scheduled in either GEMM.
    assert hip.align_map is case.slot_map
    assert hip.pairs_seen == [keep.reshape(-1).nonzero().flatten().tolist()] * 2


@pytest.mark.parametrize("deterministic", [False, True])
def test_partial_sum_identity_through_apply(monkeypatch, deterministic):
    case = Case(4, 12, global_experts=512, pool=24, masked=0.1)
    subset = [1, 2, 5, 6, 9, 10, 13, 14, 17, 18, 21, 22]
    complement = [e for e in range(24) if e not in subset]
    full, hip_full = apply_case(monkeypatch, case, deterministic=deterministic)
    part_a, hip_a = apply_case(
        monkeypatch, case.restricted(subset), deterministic=deterministic
    )
    part_b, hip_b = apply_case(
        monkeypatch, case.restricted(complement), deterministic=deterministic
    )
    assert relative_l2(part_a.double() + part_b.double(), full.double()) < 5e-3
    if deterministic:
        # Every pair adds exact integers, so splitting the experts between
        # two ranks splits the accumulator exactly ...
        total = hip_a.accumulators[-1] + hip_b.accumulators[-1]
        assert torch.equal(total, hip_full.accumulators[-1])
        # ... and the fp16 outputs agree to three fp16 roundings.
        bound = 2.0**-10 * (part_a.abs() + part_b.abs() + full.abs()).float() + 1e-6
        assert ((part_a + part_b).float() - full.float()).abs().le(bound).all()


@pytest.mark.parametrize("tokens", [3, 24])
def test_q32_mode_is_bit_identical_under_any_arrival_order(monkeypatch, tokens):
    case = Case(5, tokens, global_experts=512, pool=24, masked=0.1)
    reference, _ = apply_case(monkeypatch, case, deterministic=True)
    for seed in (1, 2, 3):
        shuffled, _ = apply_case(monkeypatch, case, deterministic=True, order_seed=seed)
        assert torch.equal(reference, shuffled)
    default_reference, _ = apply_case(monkeypatch, case)
    shuffles = [apply_case(monkeypatch, case, order_seed=seed)[0] for seed in (1, 2, 3)]
    # The stand-in really perturbs the arithmetic: fp16 accumulation is not
    # order independent, so the invariance above is not vacuous.
    assert not all(torch.equal(default_reference, s) for s in shuffles)


def test_q32_mode_poisons_only_the_token_with_a_non_finite_input(monkeypatch):
    case = Case(6, 5, global_experts=512, pool=24)
    case.x[2, 7] = float("inf")
    output, _ = apply_case(monkeypatch, case, deterministic=True)
    nan_rows = output.isnan().all(dim=1)
    assert nan_rows.tolist() == [False, False, True, False, False]
    clean = torch.tensor([0, 1, 3, 4])
    expected = case.reference()[clean]
    assert relative_l2(output[clean], expected) < 5e-3


@pytest.mark.parametrize(
    "global_experts,residents", [(512, [2, 3, 5, 8]), (24, [3, 8, 23])]
)
def test_ignore_invalid_experts_changes_neither_the_pairs_nor_the_result(
    monkeypatch, global_experts, residents
):
    """The kernel now emits local slots itself. The old Python gather mapped
    the -1 tail of expert_ids through ``expert_map[-1]``, a real slot when the
    last global expert is resident (second case); only the
    ``num_tokens_post_padded`` early exit made that harmless."""
    case = Case(
        7, 6, global_experts=global_experts, pool=24, masked=0.2, residents=residents
    )
    real_align = rdna3_moe.moe_align_block_size
    seen = []

    def spying(*args, **kwargs):
        seen.append((args, kwargs))
        return real_align(*args, **kwargs)

    monkeypatch.setattr(rdna3_moe, "moe_align_block_size", spying)
    new, hip_new = apply_case(monkeypatch, case, deterministic=True)
    assert len(seen) == 1
    args, kwargs = seen[0]
    assert kwargs == {"ignore_invalid_experts": True}
    assert args[2] == case.global_experts and args[3] is case.slot_map

    def legacy(topk_ids, block_size, num_experts, expert_map=None, **_):
        return real_align(topk_ids, block_size, num_experts, expert_map)

    monkeypatch.setattr(rdna3_moe, "moe_align_block_size", legacy)
    old, hip_old = apply_case(monkeypatch, case, deterministic=True)
    assert hip_new.events[0]["expert_map"] is True
    assert hip_old.events[0]["expert_map"] is False
    assert hip_old.pairs_seen == hip_new.pairs_seen
    assert torch.equal(new, old)


@pytest.mark.parametrize("override", [None, 1, 2, 4, 8])
@pytest.mark.parametrize("tokens", [1, 4, 5, 24])
def test_block_size_selection(monkeypatch, tokens, override):
    case = Case(8, tokens, global_experts=24, pool=24)
    expected = override or (1 if tokens <= 4 else 4)
    _, hip = apply_case(monkeypatch, case, block_m=override, use_map=False)
    sizes = {e["block_size"] for e in hip.events if e["op"] == "align"}
    sizes |= {e["block_size_m"] for e in hip.events if e["op"] == "gemm"}
    assert sizes == {expected}


@pytest.mark.parametrize(
    "tokens,expected", [(1, 1), (4, 1), (5, 4), (48, 4), (2048, 4)]
)
def test_block_size_default_choice_up_to_2048_tokens(monkeypatch, tokens, expected):
    experts = make_experts(monkeypatch, Case(0, 1))
    assert experts._block_size_m(tokens) == expected


@pytest.mark.parametrize("value", ["3", "5", "16", "-1"])
def test_invalid_block_size_override_is_rejected(monkeypatch, value):
    with pytest.raises(ValueError, match="VLLM_RDNA3_MOE_BLOCK_M"):
        make_experts(monkeypatch, Case(0, 1), block_m=value)


def test_q32_mode_rejects_bf16_but_default_mode_accepts_it(monkeypatch):
    case = Case(9, 2, global_experts=512, pool=24)
    with pytest.raises(ValueError, match="fp16"):
        apply_case(monkeypatch, case, deterministic=True, dtype=torch.bfloat16)
    output, _ = apply_case(monkeypatch, case, dtype=torch.bfloat16)
    assert relative_l2(output, case.reference()) < 5e-2


def test_expert_map_must_be_a_contiguous_int32_covering_every_global_expert(
    monkeypatch,
):
    case = Case(10, 2, global_experts=512, pool=24)
    wide = torch.stack([case.slot_map, case.slot_map], dim=1)
    for bad in (case.slot_map.to(torch.int64), wide[:, 0], case.slot_map[:100]):
        broken = copy.copy(case)
        broken.slot_map = bad
        with pytest.raises(AssertionError):
            apply_case(monkeypatch, broken)


# --- Graph-capture contract: no host reads, shapes from host ints only -----


def _noop(*args, **kwargs):
    return None


def test_fake_tensor_mode_catches_host_reads_and_data_dependent_shapes():
    with FakeTensorMode():
        ids = torch.empty(4, dtype=torch.int32)
        for read in (ids.sum().item, ids.tolist, lambda: bool(ids[0] > 0)):
            with pytest.raises(DataDependentOutputException):
                read()
        with pytest.raises(DynamicOutputShapeException):
            ids.nonzero()


@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize("tokens", [1, 3, 5, 24, 48])
def test_apply_has_no_host_read_or_data_dependent_shape(
    monkeypatch, tokens, deterministic
):
    """Hub shapes (K=2560, N=640, 10 of 512 experts, 64 slots) under
    FakeTensorMode, which rejects ``.item()`` and data-dependent shapes."""
    case = Case(0, 1, global_experts=512, pool=24)
    rocm = torch.ops._rocm_C
    monkeypatch.setattr(rocm, "moe_gptq_gemm_rdna3", _noop, raising=False)
    monkeypatch.setattr(rocm, "moe_fixedpoint_to_half_rdna3", _noop, raising=False)
    monkeypatch.setattr(ops, "moe_align_block_size", _noop)
    experts = make_experts(monkeypatch, case, deterministic=deterministic)
    experts.activation = _noop
    with FakeTensorMode():
        hidden = torch.empty(tokens, 2560, dtype=torch.float16)
        w1 = torch.empty(64, 320, 1280, dtype=torch.int32)
        w2 = torch.empty(64, 80, 2560, dtype=torch.int32)
        ids = torch.empty(tokens, TOP_K, dtype=torch.int32)
        weights = torch.empty(tokens, TOP_K, dtype=torch.float32)
        slot_map = torch.empty(512, dtype=torch.int32)
        shapes = experts.workspace_shapes(
            tokens, 640, 2560, TOP_K, 512, 64, None, MoEActivation.SILU
        )
        output = torch.empty(shapes[2], dtype=torch.float16)
        experts.apply(
            output,
            hidden,
            w1,
            w2,
            weights,
            ids,
            MoEActivation.SILU,
            512,
            slot_map,
            None,
            None,
            torch.empty(shapes[0], dtype=torch.float16),
            torch.empty(shapes[1], dtype=torch.float16),
            None,
            False,
        )


@pytest.mark.parametrize(
    "tokens,align_shapes,workspace_elements",
    [
        (1, ((10,), (10,)), 19_200),
        (48, ((1920,), (480,)), 921_600),
        (2048, ((22016,), (5504,)), 39_321_600),
    ],
)
def test_hub_shape_buffers_for_block_size_default(
    monkeypatch, tokens, align_shapes, workspace_elements
):
    """Buffer sizes at the hub shapes, as allocated by the real wrapper."""
    shapes = []

    def record(topk_ids, num_experts, block_size, sorted_ids, expert_ids, *rest):
        shapes.append((tuple(sorted_ids.shape), tuple(expert_ids.shape)))

    monkeypatch.setattr(ops, "moe_align_block_size", record)
    case = Case(0, 1, global_experts=512, pool=24)
    experts = make_experts(monkeypatch, case)
    with FakeTensorMode():
        ids = torch.empty(tokens, TOP_K, dtype=torch.int32)
        rdna3_moe.moe_align_block_size(
            ids,
            experts._block_size_m(tokens),
            512,
            torch.empty(512, dtype=torch.int32),
            ignore_invalid_experts=True,
        )
    assert shapes == [align_shapes]
    workspace = experts.workspace_shapes(
        tokens, 640, 2560, TOP_K, 512, 64, None, MoEActivation.SILU
    )
    assert workspace == ((tokens, 2560), (workspace_elements,), (tokens, 2560))


@pytest.mark.parametrize(
    "tokens,gate_up_bytes,output_bytes",
    [(48, 4_919_040, 983_424), (2048, 209_879_040, 41_959_424)],
)
def test_q32_scratch_bytes_at_hub_shapes(tokens, gate_up_bytes, output_bytes):
    gate_up = torch.empty(tokens * TOP_K, 1280, dtype=torch.float16, device="meta")
    output = torch.empty(tokens, 2560, dtype=torch.float16, device="meta")
    for destination, expected in ((gate_up, gate_up_bytes), (output, output_bytes)):
        scratch = rdna3_moe._q32_scratch(destination)
        assert scratch.dtype == torch.int64
        assert scratch.shape == (destination.size(0), destination.size(1) + 1)
        assert scratch.numel() * 8 == expected


# --- S1b: the WMMA prefill path of Rdna3WNA16Experts.apply ------------------


def expected_wmma_flow(case, tokens, *, block_size, on_input=False):
    pairs = tokens * TOP_K
    gate_up = 2 * INTER
    return [
        {
            "op": "align",
            "num_experts": case.global_experts,
            "block_size": block_size,
            "expert_map": True,
        },
        {
            "op": "wmma_gemm",
            "fixed": False,
            "a_shape": (tokens, HIDDEN),
            "c_shape": (pairs, gate_up),
            "top_k": TOP_K,
            "block_size_m": block_size,
            "mul_topk_weight": on_input,
            "output_topk": 0,
            "topk_weights": pairs if on_input else 0,
        },
        {
            "op": "activation",
            "input_shape": (pairs, gate_up),
            "output_shape": (pairs, INTER),
        },
        {
            "op": "wmma_gemm",
            "fixed": True,
            "a_shape": (pairs, INTER),
            "c_shape": (tokens, HIDDEN + 1),
            "top_k": 1,
            "block_size_m": block_size,
            "mul_topk_weight": not on_input,
            "output_topk": TOP_K,
            "topk_weights": 0 if on_input else pairs,
        },
        {
            "op": "cast",
            "accumulator_shape": (tokens, HIDDEN + 1),
            "output_shape": (tokens, HIDDEN),
        },
    ]


@pytest.mark.parametrize("on_input", [False, True])
@pytest.mark.parametrize(
    "tokens,global_experts,block_size",
    [(24, 512, 16), (40, 24, 16), (64, 24, 32), (200, 24, 64)],
)
def test_wmma_flow_stores_gemm1_directly_and_reduces_gemm2_through_q32(
    monkeypatch, tokens, global_experts, block_size, on_input
):
    case = Case(21, tokens, global_experts=global_experts, pool=24, masked=0.1)
    output, hip = apply_case(
        monkeypatch, case, wmma=True, wmma_min_m=1, on_input=on_input
    )
    assert hip.events == expected_wmma_flow(
        case, tokens, block_size=block_size, on_input=on_input
    )
    assert relative_l2(output, case.reference(on_input)) < 5e-3
    assert not bool((output == 7.0).any()), "the cast must overwrite the output"


@pytest.mark.parametrize(
    "tokens,top_k,experts,expected",
    [
        (256, 10, 512, 16),
        (1023, 10, 512, 16),
        (1024, 10, 512, 32),
        (4095, 10, 512, 32),
        (4096, 10, 512, 64),
        (16384, 10, 512, 64),
        (64, 8, 16, 32),
        (7, 1, 0, 16),
    ],
)
def test_wmma_block_size_follows_the_mean_rows_per_expert(
    monkeypatch, tokens, top_k, experts, expected
):
    wmma_experts = make_experts(monkeypatch, Case(0, 1), wmma=True)
    assert wmma_experts._wmma_block_size_m(tokens, top_k, experts) == expected


@pytest.mark.parametrize("block_m", [16, 32, 64])
def test_wmma_block_override_wins_over_the_heuristic(monkeypatch, block_m):
    case = Case(22, 24, global_experts=512, pool=24, masked=0.1)
    output, hip = apply_case(
        monkeypatch, case, wmma=True, wmma_min_m=1, wmma_block_m=block_m
    )
    assert hip.events == expected_wmma_flow(case, 24, block_size=block_m)
    assert relative_l2(output, case.reference()) < 5e-3


@pytest.mark.parametrize("value", ["1", "8", "48", "128", "-16"])
def test_invalid_wmma_block_override_is_rejected(monkeypatch, value):
    for wmma in (True, False):
        with pytest.raises(ValueError, match="VLLM_RDNA3_MOE_WMMA_BLOCK_M"):
            make_experts(monkeypatch, Case(0, 1), wmma=wmma, wmma_block_m=value)


def test_wmma_enabled_without_the_op_fails_at_construction(monkeypatch):
    with pytest.raises(RuntimeError, match="moe_gptq_gemm_rdna3_wmma"):
        make_experts(monkeypatch, Case(0, 1), wmma=True, wmma_op=False)
    make_experts(monkeypatch, Case(0, 1), wmma=False, wmma_op=False)


def test_wmma_threshold_keeps_small_batches_on_the_scalar_kernel(monkeypatch):
    case = Case(23, 24, global_experts=512, pool=24, masked=0.1)
    below, hip_below = apply_case(monkeypatch, case, wmma=True, wmma_min_m=25)
    assert hip_below.events == expected_flow(case, 24, fixed=False, block_size=4)
    at, hip_at = apply_case(monkeypatch, case, wmma=True, wmma_min_m=24)
    assert [event["op"] for event in hip_at.events] == [
        "align",
        "wmma_gemm",
        "activation",
        "wmma_gemm",
        "cast",
    ]
    # The threshold alone does not enable the path.
    off, hip_off = apply_case(monkeypatch, case, wmma=False, wmma_min_m=1)
    assert hip_off.events == hip_below.events
    for result in (below, at, off):
        assert relative_l2(result, case.reference()) < 5e-3


def test_wmma_needs_fp16_activations_at_any_batch_size(monkeypatch):
    case = Case(24, 4, global_experts=512, pool=24)
    for min_m in (1, 10**6):
        with pytest.raises(ValueError, match="fp16"):
            apply_case(
                monkeypatch, case, wmma=True, wmma_min_m=min_m, dtype=torch.bfloat16
            )
    output, _ = apply_case(monkeypatch, case, wmma=False, dtype=torch.bfloat16)
    assert relative_l2(output, case.reference()) < 5e-2


@pytest.mark.parametrize("deterministic", [False, True])
def test_wmma_path_ignores_the_deterministic_flag_and_any_arrival_order(
    monkeypatch, deterministic
):
    case = Case(25, 24, global_experts=512, pool=24, masked=0.1)
    reference, hip = apply_case(monkeypatch, case, wmma=True, wmma_min_m=1)
    flagged, hip_flagged = apply_case(
        monkeypatch, case, wmma=True, wmma_min_m=1, deterministic=deterministic
    )
    assert hip_flagged.events == hip.events
    assert torch.equal(reference, flagged)
    for seed in (1, 2, 3):
        shuffled, _ = apply_case(
            monkeypatch,
            case,
            wmma=True,
            wmma_min_m=1,
            deterministic=deterministic,
            order_seed=seed,
        )
        assert torch.equal(reference, shuffled)


@pytest.mark.parametrize("tokens", [3, 24])
def test_wmma_hub_contract_masked_ids_and_slot_map(monkeypatch, tokens):
    residents = [0, 3, 4, 8, 9, 15, 16, 22, 30, 33, 36, 39]
    case = Case(
        27, tokens, global_experts=512, pool=40, masked=0.25, residents=residents
    )
    keep = (case.ids >= 0) & (case.slot_map[case.ids.clamp(min=0).long()] >= 0)
    assert 0 < keep.sum() < keep.numel()
    output, hip = apply_case(monkeypatch, case, wmma=True, wmma_min_m=1)
    # Rows of dropped pairs keep the workspace garbage after GEMM 1 (it never
    # zeroes) and must not leak into the result.
    assert relative_l2(output, case.reference()) < 5e-3
    assert hip.align_map is case.slot_map
    assert hip.pairs_seen == [keep.reshape(-1).nonzero().flatten().tolist()] * 2


def test_wmma_partial_sum_identity_through_apply(monkeypatch):
    case = Case(28, 12, global_experts=512, pool=24, masked=0.1)
    subset = [1, 2, 5, 6, 9, 10, 13, 14, 17, 18, 21, 22]
    complement = [e for e in range(24) if e not in subset]
    full, hip_full = apply_case(monkeypatch, case, wmma=True, wmma_min_m=1)
    part_a, hip_a = apply_case(
        monkeypatch, case.restricted(subset), wmma=True, wmma_min_m=1
    )
    part_b, hip_b = apply_case(
        monkeypatch, case.restricted(complement), wmma=True, wmma_min_m=1
    )
    assert relative_l2(part_a.double() + part_b.double(), full.double()) < 5e-3
    # Q32 adds are exact: two ranks split the accumulator exactly.
    total = hip_a.accumulators[-1] + hip_b.accumulators[-1]
    assert torch.equal(total, hip_full.accumulators[-1])


@pytest.mark.parametrize("tokens", [256, 1024, 2048])
def test_wmma_apply_has_no_host_read_or_data_dependent_shape(monkeypatch, tokens):
    """Hub shapes under FakeTensorMode (graph-capture contract of the path)."""
    case = Case(0, 1, global_experts=512, pool=24)
    rocm = torch.ops._rocm_C
    monkeypatch.setattr(ops, "moe_gptq_gemm_rdna3_wmma", _noop, raising=False)
    monkeypatch.setattr(rocm, "moe_fixedpoint_to_half_rdna3", _noop, raising=False)
    monkeypatch.setattr(ops, "moe_align_block_size", _noop)
    experts = make_experts(monkeypatch, case, wmma=True, wmma_min_m=256)
    experts.activation = _noop
    with FakeTensorMode():
        hidden = torch.empty(tokens, 2560, dtype=torch.float16)
        w1 = torch.empty(64, 320, 1280, dtype=torch.int32)
        w2 = torch.empty(64, 80, 2560, dtype=torch.int32)
        ids = torch.empty(tokens, TOP_K, dtype=torch.int32)
        weights = torch.empty(tokens, TOP_K, dtype=torch.float32)
        slot_map = torch.empty(512, dtype=torch.int32)
        shapes = experts.workspace_shapes(
            tokens, 640, 2560, TOP_K, 512, 64, None, MoEActivation.SILU
        )
        output = torch.empty(shapes[2], dtype=torch.float16)
        experts.apply(
            output,
            hidden,
            w1,
            w2,
            weights,
            ids,
            MoEActivation.SILU,
            512,
            slot_map,
            None,
            None,
            torch.empty(shapes[0], dtype=torch.float16),
            torch.empty(shapes[1], dtype=torch.float16),
            None,
            False,
        )
