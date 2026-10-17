# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""gfx1030 (RDNA2) HIP top-k/top-p masking vs the reference implementations.

The HIP kernel in ``csrc/rocm/v1_topk_topp_rdna2.cu`` replaces the Triton
kernels of ``vllm/v1/sample/ops/topk_topp_triton.py`` (the monolithic
``_topk_topp_kernel`` and the ``_topp_sb_*`` split pipeline).  ``apply_top_k_top_p``
dispatches to it unconditionally on gfx10x.

Reference semantics: ``apply_top_k_top_p_pytorch`` -- top-k keeps every logit
>= the k-th largest, then top-p keeps the smallest set of the survivors whose
normalised mass reaches p.  Exact ties at the top-p boundary are the one place
where the references disagree with each other (pytorch keeps the highest-index
copies of the tied group, the Triton pipeline the lowest); the HIP kernel keeps
the whole tied group, which is a superset of both and has no effect on fp32
serving logits, so tie cases are asserted with an invariant instead of set
equality.
"""

import pytest
import torch

from vllm.platforms.rdna import on_rdna2
from vllm.v1.sample.ops import topk_topp_sampler as tt

pytestmark = pytest.mark.skipif(not on_rdna2(), reason="gfx10x HIP kernels only")

NEG = float("-inf")
DEV = torch.device("cuda")
SERVED_VOCAB = 151936


@pytest.fixture(autouse=True)
def _hip_binding_available():
    # gfx10x must raise rather than silently fall back to Triton.
    assert callable(tt._rdna2_topk_topp_op())


def _kept(masked, mask_value=NEG):
    return [
        frozenset(torch.nonzero(row > mask_value).flatten().tolist())
        for row in masked
    ]


def _hip(logits, k, p, mask_value=NEG):
    return tt.apply_top_k_top_p_rdna2(logits.clone(), k, p, mask_value)


def _ref(logits, k, p):
    return tt.apply_top_k_top_p_pytorch(logits.clone(), k, p)


def _triton(logits, k, p):
    return tt.apply_top_k_top_p_triton(logits.clone(), k, p)


def _idx(values, dtype=torch.int32):
    return torch.tensor(values, dtype=dtype, device=DEV)


def _check(logits, k, p):
    """HIP == pytorch exactly; HIP == triton whenever triton == pytorch."""
    hip = _kept(_hip(logits, k, p))
    ref = _kept(_ref(logits, k, p))
    assert hip == ref
    if tt.HAS_TRITON:
        tri = _kept(_triton(logits, k, p))
        if tri == ref:
            assert hip == tri
    return hip


def test_served_shape_matches_references():
    """8 requests x MTP2 rows, vocab 151936, the deployed generation config."""
    torch.manual_seed(0)
    logits = torch.randn(24, SERVED_VOCAB, dtype=torch.float32, device=DEV) * 3.0
    k = _idx([20] * 24)
    p = torch.full((24,), 0.95, dtype=torch.float32, device=DEV)
    hip = _kept(_hip(logits, k, p))
    ref = _kept(_ref(logits, k, p))
    assert hip == ref
    assert max(len(s) for s in hip) <= 20  # top-k is a floor, not an exact count
    assert min(len(s) for s in hip) >= 1
    if tt.HAS_TRITON:
        # Triton enforces exactly k here; the same 20 logits must survive.
        assert hip == _kept(_triton(logits, k, p))


def test_sampling_workflow_shape():
    """k=50 / p=0.9 is what SamplingParams.for_sampler_warmup() exercises."""
    torch.manual_seed(1)
    logits = torch.randn(24, SERVED_VOCAB, dtype=torch.float32, device=DEV)
    _check(logits, _idx([50] * 24), torch.full((24,), 0.9, dtype=torch.float32, device=DEV))


@pytest.mark.parametrize(
    ("rows", "vocab", "k", "p"),
    [
        (0, 1032, [], []),  # zero rows
        (1, 1032, [1], [0.95]),  # one row, k = 1
        (1, 1032, [1], [1e-9]),  # p very small -> only the argmax survives
        (3, 1026, [1, 500, 1026], [0.5, 0.95, 1.0]),  # vocab % 4 != 0 (scalar path)
        (3, 4100, [1, 100, 4100], [0.9, 0.95, 0.999]),  # vocab not a block multiple
        (4, 4096, [4096, 4096, 4096, 4096], [0.95] * 4),  # k == vocab: no top-k
        (4, 1032, [20, 20, 1032, 1032], [1.0, 0.95, 1.0, 0.95]),  # mixed rows
        (4, 1032, [1032, 1032, 20, 20], [0.95, 0.95, 1.0, 1.0]),  # top-k only
        (2, 1032, [1032, 7], [0.5, 0.999999]),
    ],
)
def test_edge_cases_match_references(rows, vocab, k, p):
    torch.manual_seed(2)
    logits = torch.randn(rows, vocab, dtype=torch.float32, device=DEV) * 3.0
    _check(logits, _idx(k), _idx(p, torch.float32))


def test_no_filters_returns_input():
    torch.manual_seed(3)
    logits = torch.randn(4, 1032, dtype=torch.float32, device=DEV)
    for k, p in ((None, None), (_idx([1032] * 4), None), (None, torch.ones(4, device=DEV))):
        out = tt.apply_top_k_top_p(logits.clone(), k, p)
        assert torch.equal(out, logits) or _kept(out) == _kept(logits)


def test_fewer_finite_than_k_keeps_all_finite():
    """A grammar mask can leave fewer finite logits than k."""
    torch.manual_seed(4)
    logits = torch.randn(2, 1032, dtype=torch.float32, device=DEV)
    logits[0, 10:] = NEG
    logits[1, :900] = NEG
    _check(logits, _idx([20, 20]), _idx([0.9, 0.9], torch.float32))
    # top-k alone keeps every finite logit when there are fewer than k of them.
    assert _kept(_hip(logits, _idx([20, 20]), None))[0] == frozenset(range(10))


def test_grammar_masked_rows_match_references():
    torch.manual_seed(5)
    logits = torch.randn(4, 151936, dtype=torch.float32, device=DEV) * 2.0
    logits[0, 1000:] = NEG
    logits[1, :150000] = NEG
    logits[3, ::2] = NEG
    _check(logits, _idx([20] * 4), torch.full((4,), 0.9, dtype=torch.float32, device=DEV))


def test_mask_value_is_applied():
    torch.manual_seed(6)
    logits = torch.randn(3, 1032, dtype=torch.float32, device=DEV)
    k, p = _idx([20] * 3), _idx([0.9] * 3, torch.float32)
    masked = _hip(logits, k, p, mask_value=-1e30)
    assert _kept(masked, -1e30) == _kept(_ref(logits, k, p))
    dropped = masked[0] == -1e30
    assert dropped.any()
    assert torch.equal(masked[0][~dropped], logits[0][~dropped])


def test_topk_ties_match_pytorch_exactly():
    """top-k alone keeps every tied copy in both implementations."""
    logits = torch.zeros(2, 1032, dtype=torch.float32, device=DEV)
    logits[0] = torch.arange(1032, dtype=torch.float32, device=DEV)
    logits[0, 100:] = 0.0  # many ties, all far below the top-k boundary
    logits[1, :200] = 3.0  # a tie group covering the whole top-k set
    k = _idx([20, 50])
    assert _kept(_hip(logits, k, None)) == _kept(_ref(logits, k, None))


def test_topp_boundary_ties_keep_the_whole_group():
    """Documented divergence: HIP keeps the tied boundary group in full."""
    logits = torch.ones(2, 1032, dtype=torch.float32, device=DEV)
    k, p = _idx([50, 50]), _idx([0.95, 0.95], torch.float32)
    hip, ref = _kept(_hip(logits, k, p)), _kept(_ref(logits, k, p))
    assert ref <= hip  # superset
    for row in range(2):
        extra = hip[row] - ref[row]
        boundary = min(logits[row][i].item() for i in ref[row])
        assert all(logits[row][i].item() == boundary for i in extra)
        # nothing below the boundary survives
        assert all(logits[row][i].item() >= boundary for i in hip[row])


def test_hip_path_taken_when_triton_is_poisoned(monkeypatch):
    """The gfx10x dispatch must not reach the Triton kernels."""

    def _boom(*args, **kwargs):
        raise AssertionError("Triton top-k/top-p kernel was called on gfx10x")

    monkeypatch.setattr(tt, "apply_top_k_top_p_triton", _boom)
    torch.manual_seed(7)
    logits = torch.randn(4, 1032, dtype=torch.float32, device=DEV)
    k, p = _idx([20] * 4), _idx([0.9] * 4, torch.float32)
    assert _kept(tt.apply_top_k_top_p(logits.clone(), k, p)) == _kept(_ref(logits, k, p))


def test_missing_hip_binding_raises(monkeypatch):
    """A build without the HIP binding must fail loudly, not fall back."""

    class _NoOp:
        def __getattr__(self, name):
            raise AttributeError(name)

    class _NoNamespace:
        _rocm_C = _NoOp()

    class _NoTorch:
        ops = _NoNamespace()

    monkeypatch.setattr(tt, "torch", _NoTorch)
    with pytest.raises(RuntimeError, match="topk_topp_rdna2"):
        tt._rdna2_topk_topp_op()


@pytest.mark.parametrize("seed", [11, 12, 13])
def test_randomized_vocab_and_params(seed):
    torch.manual_seed(seed)
    rows, vocab = 9, 15104
    logits = torch.randn(rows, vocab, dtype=torch.float32, device=DEV) * 3.0
    k = _idx([1, 2, 20, 50, 100, vocab, vocab, 20, 5])
    p = _idx([0.0, 1e-6, 0.05, 0.5, 0.95, 0.999, 1.0, 0.95, 0.9], torch.float32)
    _check(logits, k, p)
