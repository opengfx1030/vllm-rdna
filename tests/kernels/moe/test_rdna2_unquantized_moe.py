# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Parity and dispatch tests for the RDNA2 unquantized (fp16) MoE backend.

Covers ``csrc/rocm/moe_unquantized_rdna2.cu`` through
``Rdna2UnquantizedExperts.apply`` (the deployed entry point) on the MTP
drafter's real geometry: E=128 local / 512 global, N=640, K=2560, top_k=10,
M=8..24, fp16, EP4.
"""

import pytest
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.config.kernel import MoEBackend  # noqa: F401
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.experts.rdna2_unquantized_moe import (
    Rdna2UnquantizedExperts,
    rdna2_unquantized_moe_available,
)
from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
    UnquantizedMoeBackend,
    _get_priority_backends,
    backend_to_kernel_cls,
)
from vllm.model_executor.layers.fused_moe.utils import _resize_cache
from vllm.platforms.rdna import on_rdna2

pytestmark = pytest.mark.skipif(
    not on_rdna2(), reason="RDNA2 (gfx10x) HIP path only"
)

# Real serving geometry (Qwen4ExpMTP drafter).
HIDDEN = 2560
INTER = 640
E_LOCAL = 128
E_GLOBAL = 512
TOPK = 10

# fp16 has ~5e-4 relative resolution and the down projection rounds once per
# (token, top-k) slot; activations reach ~0.5 and outputs ~0.1 here.
ATOL = 1e-3
RTOL = 2e-2


def _experts():
    moe_config = FusedMoEConfig(
        num_experts=E_GLOBAL,
        experts_per_token=TOPK,
        hidden_dim=HIDDEN,
        intermediate_size=INTER,
        num_local_experts=E_LOCAL,
        num_logical_experts=E_GLOBAL,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        activation=MoEActivation.SILU,
        in_dtype=torch.float16,
        device="cuda",
        routing_method=RoutingMethodType.Default,
        has_bias=False,
    )
    return Rdna2UnquantizedExperts(moe_config, FusedMoEQuantConfig.make())


def _inputs(m, hidden=HIDDEN, inter=INTER, e_local=E_LOCAL, e_global=E_GLOBAL,
            topk=TOPK, with_map=True, device="cuda", seed=0):
    torch.manual_seed(seed)
    x = (torch.randn(m, hidden, device=device, dtype=torch.float16) * 0.1)
    w13 = (torch.randn(e_local, 2 * inter, hidden, device=device,
                       dtype=torch.float16) * 0.05)
    w2 = (torch.randn(e_local, hidden, inter, device=device,
                      dtype=torch.float16) * 0.05)
    ids = torch.randint(
        0, e_local if not with_map else e_global, (m, topk), device=device,
        dtype=torch.int32,
    )
    w = torch.rand(m, topk, device=device, dtype=torch.float32)
    w /= w.sum(-1, keepdim=True)
    emap = None
    if with_map:
        emap = torch.full((e_global,), -1, dtype=torch.int32, device=device)
        emap[:e_local] = torch.arange(e_local, dtype=torch.int32, device=device)
    return x, w13, w2, ids, w, emap


def _run(experts, x, w13, w2, ids, w, emap, mul_weight=True,
         apply_router_weight_on_input=False, alias_output=False, ws2=None):
    m = x.shape[0]
    topk = ids.shape[1]
    ws13 = torch.empty((m, topk, max(w13.shape[1] // 2, x.shape[1])),
                       device=x.device, dtype=torch.float16)
    if alias_output:
        # Exactly what the modular runner passes: workspace13 is the whole
        # common_workspace and fused_out is its first M*K elements
        # (modular_kernel.py:_allocate_buffers -> get_simultaneous/fused_out).
        out = ws13.flatten()[: m * x.shape[1]].view(m, x.shape[1])
    else:
        out = torch.empty(m, x.shape[1], device=x.device, dtype=torch.float16)
    if ws2 is None:
        ws2 = torch.empty((m, topk, max(w13.shape[1] // 2, x.shape[1])),
                          device=x.device, dtype=torch.float16)
    experts.apply(
        output=out,
        hidden_states=x,
        w1=w13,
        w2=w2,
        topk_weights=w,
        topk_ids=ids,
        activation=MoEActivation.SILU,
        global_num_experts=E_GLOBAL,
        expert_map=emap,
        a1q_scale=None,
        a2_scale=None,
        workspace13=ws13,
        workspace2=ws2,
        expert_tokens_meta=None,
        apply_router_weight_on_input=apply_router_weight_on_input,
    )
    return out, ws13


def _reference(x, w13, w2, ids, w, emap, mul_weight=True):
    m, inter = x.shape[0], w2.shape[2]
    out = torch.zeros(m, x.shape[1], device=x.device, dtype=torch.float32)
    for i in range(m):
        xi = x[i].float()
        for s in range(ids.shape[1]):
            g = int(ids[i, s])
            local = g if emap is None else int(emap[g])
            if local < 0:
                continue
            gate = w13[local, :inter].float() @ xi
            up = w13[local, inter:].float() @ xi
            act = torch.nn.functional.silu(gate) * up
            weight = float(w[i, s]) if mul_weight else 1.0
            out[i] += weight * (w2[local].float() @ act)
    return out


@pytest.mark.parametrize("m", [1, 3, 8, 24])
def test_hip_matches_fp32_reference(m):
    experts = _experts()
    x, w13, w2, ids, w, emap = _inputs(m)
    out, act = _run(experts, x, w13, w2, ids, w, emap)
    ref = _reference(x, w13, w2, ids, w, emap)
    torch.testing.assert_close(out.float(), ref, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("m", [1, 5, 24])
def test_hip_without_expert_map(m):
    """No EP: every global id is local."""
    experts = _experts()
    x, w13, w2, ids, w, _ = _inputs(m, with_map=False)
    out, _ = _run(experts, x, w13, w2, ids, w, None)
    ref = _reference(x, w13, w2, ids, w, None)
    torch.testing.assert_close(out.float(), ref, atol=ATOL, rtol=RTOL)


def test_hip_matches_no_expert_map_when_all_routed_experts_are_remote():
    """Every routed expert is non-local -> the all-reduce of the EP ranks
    leaves the rank-local contribution at zero, not uninitialized."""
    experts = _experts()
    x, w13, w2, ids, w, _ = _inputs(8)
    emap = torch.full((E_GLOBAL,), -1, dtype=torch.int32, device=x.device)
    out, _ = _run(experts, x, w13, w2, ids, w, emap)
    assert torch.equal(out, torch.zeros_like(out))


def test_router_weight_on_input_is_not_applied_twice():
    experts = _experts()
    x, w13, w2, ids, w, emap = _inputs(8)
    out, _ = _run(experts, x, w13, w2, ids, w, emap,
                  apply_router_weight_on_input=True)
    ref = _reference(x, w13, w2, ids, w, emap, mul_weight=False)
    torch.testing.assert_close(out.float(), ref, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize(
    ("hidden", "inter", "e_local", "e_global", "topk"),
    [(2560, 640, 128, 512, 10), (64, 8, 4, 8, 2), (128, 16, 4, 4, 1),
     (256, 24, 6, 6, 3)],
)
def test_edge_shapes(hidden, inter, e_local, e_global, topk):
    """Odd tile sizes (N or K not a multiple of the block tile)."""
    experts = _experts()
    x, w13, w2, ids, w, _ = _inputs(4, hidden, inter, e_local, e_global, topk,
                                    with_map=False)
    out, _ = _run(experts, x, w13, w2, ids, w, None)
    ref = _reference(x, w13, w2, ids, w, None)
    torch.testing.assert_close(out.float(), ref, atol=ATOL, rtol=RTOL)


def test_activation_cache_is_written():
    """The gate/up stage writes every routed pair of workspace13."""
    experts = _experts()
    x, w13, w2, ids, w, emap = _inputs(4)
    _, ws13 = _run(experts, x, w13, w2, ids, w, emap)
    n = INTER
    a = _resize_cache(ws13, (x.shape[0] * TOPK, n)).float()
    ref = torch.zeros_like(a)
    for i in range(x.shape[0]):
        xi = x[i].float()
        for s in range(TOPK):
            local = int(emap[int(ids[i, s])])
            if local < 0:
                continue
            gate = w13[local, :n].float() @ xi
            up = w13[local, n:].float() @ xi
            ref[i * TOPK + s] = torch.nn.functional.silu(gate) * up
    # Only routed pairs are written; the rest of the buffer is untouched.
    mask = torch.tensor(
        [int(emap[int(ids[i, s])]) >= 0
         for i in range(x.shape[0]) for s in range(TOPK)],
        device=a.device,
    )
    torch.testing.assert_close(a[mask], ref[mask], atol=1e-3, rtol=2e-2)


@pytest.mark.skipif(not rdna2_unquantized_moe_available(), reason="no HIP op")
def test_matches_triton_fused_experts_on_serving_shape():
    """Direct A/B against the Triton fused_moe_kernel path being replaced."""
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

    experts = _experts()
    x, w13, w2, ids, w, emap = _inputs(8)
    ref = fused_experts(
        x, w13, w2, w, ids, activation=MoEActivation.SILU,
        global_num_experts=E_GLOBAL, expert_map=emap,
        quant_config=FusedMoEQuantConfig.make(),
    )
    out, _ = _run(experts, x, w13, w2, ids, w, emap)
    torch.testing.assert_close(out.float(), ref.float(), atol=ATOL, rtol=RTOL)



@pytest.mark.skipif(not rdna2_unquantized_moe_available(), reason="no HIP op")
def test_down_rounds_each_slot_to_fp16_and_sums_topk_in_fp32():
    """One fp16 rounding per (token, top-k) slot, folded with a fp32 sum.

    That is Triton's dataflow (`intermediate_cache3` is fp16, `moe_sum`
    accumulates in fp32); accumulating the routed contributions in fp16
    instead shifts every draft logit and collapses MTP acceptance.
    """
    experts = _experts()
    x, w13, w2, ids, w, emap = _inputs(8)
    # Zeroed: a (token, slot) routed to a remote expert is never written, and
    # the pad-aware reduction must contribute nothing for it.
    ws2 = torch.zeros(
        (x.shape[0], TOPK, HIDDEN), device=x.device, dtype=torch.float16
    )
    out, _ = _run(experts, x, w13, w2, ids, w, emap, ws2=ws2)

    acc = torch.zeros_like(out, dtype=torch.float32)
    for s in range(TOPK):
        acc = acc + ws2[:, s].float()
    assert torch.equal(out, acc.half())


@pytest.mark.skipif(not rdna2_unquantized_moe_available(), reason="no HIP op")
def test_served_workspace_aliasing_does_not_corrupt_the_activations():
    """The runner makes `output` a view of `workspace13`.

    `_allocate_buffers` hands the experts `workspace13 = common_workspace`
    (its largest shape) and `fused_out = common_workspace[: M*K]`, so a down
    stage that writes `output` copies the activations it is still reading.
    """
    experts = _experts()
    x, w13, w2, ids, w, emap = _inputs(8)
    separate, _ = _run(experts, x, w13, w2, ids, w, emap)
    aliased, _ = _run(experts, x, w13, w2, ids, w, emap, alias_output=True)
    assert torch.equal(aliased, separate)
    torch.testing.assert_close(
        aliased.float(), _reference(x, w13, w2, ids, w, emap), atol=ATOL, rtol=RTOL
    )

class _PoisonedTriton:
    def __getitem__(self, grid):
        raise AssertionError(f"Triton fused_moe_kernel launched, grid {grid}")

    def __call__(self, *args, **kwargs):
        raise AssertionError("Triton fused_moe_kernel launched")


def test_hip_path_is_taken_when_triton_is_poisoned(monkeypatch):
    import vllm.model_executor.layers.fused_moe.fused_moe as fm

    monkeypatch.setattr(fm, "fused_moe_kernel", _PoisonedTriton())
    monkeypatch.setattr(fm, "invoke_fused_moe_triton_kernel", _PoisonedTriton())
    experts = _experts()
    x, w13, w2, ids, w, emap = _inputs(8)
    out, _ = _run(experts, x, w13, w2, ids, w, emap)
    ref = _reference(x, w13, w2, ids, w, emap)
    torch.testing.assert_close(out.float(), ref, atol=ATOL, rtol=RTOL)


def test_oracle_prefers_rdna2_hip():
    assert _get_priority_backends(None)[0] == UnquantizedMoeBackend.RDNA2_HIP
    assert backend_to_kernel_cls(UnquantizedMoeBackend.RDNA2_HIP) == [
        Rdna2UnquantizedExperts
    ]


def test_missing_binding_raises(monkeypatch):
    import vllm.model_executor.layers.fused_moe.experts.rdna2_unquantized_moe as m

    monkeypatch.setattr(m, "rdna2_unquantized_moe_available", lambda: False)
    with pytest.raises(RuntimeError, match="refusing to fall back"):
        m._require_ops()
