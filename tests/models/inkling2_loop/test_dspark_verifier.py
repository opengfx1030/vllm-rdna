# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact-probability gate for the verification of Inkling-2 DSpark drafts.

The DSpark drafter proposes a block of ``K`` tokens, each drawn from a draft
distribution ``q_j(. | previous token)`` (the Markov head), and the target verifies the
block with the standard speculative rejection rule: accept draft ``d_j`` with
probability ``min(1, p_j(d_j) / q_j(d_j))``, resample the first rejected position from
``norm(max(p_j - q_j, 0))`` and draw a bonus token from ``p_K`` when all ``K`` drafts
are accepted.  The rule is lossless exactly when the sampler sees the same ``q`` that
the draft sampled from, which is why the draft logits cache stores the logits the draft
sampled from (fp16 here, see ``Inkling2DSparkModel.draft_logits_spec``).

On a tiny vocabulary the output law of one verification round can be enumerated:

* ``brute_force_emitted`` walks the whole sample space (every draft block, every
  accept / reject path, every resample) with ``fractions.Fraction`` arithmetic;
* ``exact_prefix_probability`` is the closed form it must equal,
  ``P(e_0..e_{i-1} emitted, n >= i)
  = prod_{j<i-1} min(q_j, p_j)(e_j) * p_{i-1}(e_{i-1})``;
* the production rejection sampler (HIP kernels on gfx1030) is then run on ``N`` random
  blocks and the observed prefix counts are compared with the exact probabilities by a
  chi-square test (every level ``i`` is a multinomial over ``V**i`` prefixes plus the
  "fewer than ``i`` tokens" bin).

A gate that cannot fail proves nothing, so ``test_gate_rejects_biased_rules`` runs the
same statistic on deliberately wrong rules (always accept, resample from ``p`` instead
of the residual, accept with ``sqrt(p / q)``, reuse the draft's token) and requires
every one of them to exceed the threshold while the correct rule stays under it.
"""

import itertools
import random
from fractions import Fraction

import numpy as np
import pytest
import torch

VOCAB = 4
STEPS = 3  # draft tokens per block
NUM_TRIALS = 400_000
SIGMAS = 6.0  # chi-square threshold: df + SIGMAS * sqrt(2 df)
MIN_EXPECTED = 5.0


# ---------------------------------------------------------------------------
# the exact oracle
# ---------------------------------------------------------------------------
class Chain:
    """Position-dependent Markov target ``p`` and draft ``q`` over a tiny vocabulary.

    ``p[j][prev]`` is the target distribution of the token at block position ``j``
    (``0..STEPS``) given the previous token, ``q[j][prev]`` the draft distribution of
    the draft token at position ``j`` (``0..STEPS-1``).  Rows are exact fractions.
    """

    def __init__(self, p: np.ndarray, q: np.ndarray):
        assert p.shape == (STEPS + 1, VOCAB, VOCAB) and q.shape == (STEPS, VOCAB, VOCAB)
        self.p = [[self._row(r) for r in layer] for layer in p]
        self.q = [[self._row(r) for r in layer] for layer in q]

    @staticmethod
    def _row(values: np.ndarray) -> list[Fraction]:
        fr = [Fraction(float(v)) for v in values]
        total = sum(fr)
        return [v / total for v in fr]


def random_chain(seed: int) -> Chain:
    """Integer-weight rows (so the draft disagrees with the target a lot)."""
    rng = random.Random(seed)
    p = np.array(
        [
            [[rng.randint(2, 20) for _ in range(VOCAB)] for _ in range(VOCAB)]
            for _ in range(STEPS + 1)
        ],
        dtype=np.float64,
    )
    q = np.array(
        [
            [[rng.randint(1, 20) for _ in range(VOCAB)] for _ in range(VOCAB)]
            for _ in range(STEPS)
        ],
        dtype=np.float64,
    )
    p /= p.sum(-1, keepdims=True)
    q /= q.sum(-1, keepdims=True)
    return Chain(p, q)


def exact_prefix_probability(chain: Chain, prefix: tuple[int, ...]) -> Fraction:
    """``P(first len(prefix) emitted tokens == prefix, round emits >= len(prefix))``.

    The anchor token (the last committed token) is token 0.
    """
    mass, prev = Fraction(1), 0
    for j, tok in enumerate(prefix):
        p, q = chain.p[j][prev][tok], None
        if j < len(prefix) - 1:
            q = chain.q[j][prev][tok]
            mass *= min(p, q)  # the draft token was proposed and accepted
        else:
            mass *= p  # accepted draft, resample or bonus: all together ~ p
        prev = tok
    return mass


def brute_force_emitted(chain: Chain) -> dict[tuple[int, ...], Fraction]:
    """The law of the emitted token sequence, by enumerating the sample space."""
    out: dict[tuple[int, ...], Fraction] = {}

    def add(seq: tuple[int, ...], prob: Fraction) -> None:
        if prob:
            out[seq] = out.get(seq, Fraction(0)) + prob

    for block in itertools.product(range(VOCAB), repeat=STEPS):
        weight, prev = Fraction(1), 0
        for j, tok in enumerate(block):  # q-probability of the whole block
            weight *= chain.q[j][prev][tok]
            prev = tok
        alive, prev = weight, 0
        for j, tok in enumerate(block):
            p_row, q_row = chain.p[j][prev], chain.q[j][prev]
            accept = min(Fraction(1), p_row[tok] / q_row[tok])
            residual = [max(p - q, Fraction(0)) for p, q in zip(p_row, q_row)]
            total = sum(residual)
            if total:  # a rejection at j resamples from the normalised residual
                for x, r in enumerate(residual):
                    add(block[:j] + (x,), alive * (1 - accept) * r / total)
            alive *= accept
            prev = tok
        for x, p in enumerate(chain.p[STEPS][prev]):  # all accepted: the bonus
            add(block + (x,), alive * p)
    return out


def prefix_probabilities(chain: Chain, level: int) -> np.ndarray:
    """``[V**level + 1]`` exact probabilities: every prefix, then "fewer tokens"."""
    probs = np.zeros(VOCAB**level + 1)
    for prefix in itertools.product(range(VOCAB), repeat=level):
        index = int(np.ravel_multi_index(prefix, (VOCAB,) * level))
        probs[index] = float(exact_prefix_probability(chain, prefix))
    probs[-1] = 1.0 - probs[:-1].sum()
    return probs


def prefix_counts(
    sampled: np.ndarray, num_sampled: np.ndarray, level: int
) -> np.ndarray:
    """Observed counterpart of ``prefix_probabilities``."""
    long_enough = num_sampled >= level
    index = np.full(sampled.shape[0], VOCAB**level, dtype=np.int64)
    if level:
        index[long_enough] = np.ravel_multi_index(
            tuple(sampled[long_enough, j] for j in range(level)), (VOCAB,) * level
        )
    return np.bincount(index, minlength=VOCAB**level + 1)


def chi_square(counts: np.ndarray, probs: np.ndarray) -> tuple[float, int, float]:
    """(statistic, degrees of freedom, threshold); bins below ``MIN_EXPECTED`` merge."""
    n = counts.sum()
    expected = probs * n
    keep = expected >= MIN_EXPECTED
    obs, exp = counts[keep].astype(np.float64), expected[keep]
    if (~keep).any() and expected[~keep].sum() >= MIN_EXPECTED:
        obs = np.append(obs, counts[~keep].sum())
        exp = np.append(exp, expected[~keep].sum())
    stat = float(((obs - exp) ** 2 / exp).sum())
    df = len(exp) - 1
    return stat, df, df + SIGMAS * (2 * df) ** 0.5


def gate(sampled: np.ndarray, num_sampled: np.ndarray, chain: Chain) -> list[str]:
    """Every level's chi-square; the failures as readable strings."""
    failures = []
    for level in range(1, STEPS + 2):
        stat, df, limit = chi_square(
            prefix_counts(sampled, num_sampled, level),
            prefix_probabilities(chain, level),
        )
        if not stat < limit:
            failures.append(f"level {level}: chi2 {stat:.1f} >= {limit:.1f} (df {df})")
    return failures


# ---------------------------------------------------------------------------
# a vectorised model of the rule, used for the negative controls
# ---------------------------------------------------------------------------
def _tables(chain: Chain) -> tuple[torch.Tensor, torch.Tensor]:
    p = torch.tensor(
        [[[float(x) for x in r] for r in layer] for layer in chain.p],
        dtype=torch.float64,
    )
    q = torch.tensor(
        [[[float(x) for x in r] for r in layer] for layer in chain.q],
        dtype=torch.float64,
    )
    return p, q


def draw_blocks(
    p: torch.Tensor, q: torch.Tensor, n: int, gen: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draft blocks ``[n, STEPS]`` sampled from ``q`` and the ``q`` rows used."""
    prev = torch.zeros(n, dtype=torch.long, device=q.device)
    tokens = torch.empty(n, STEPS, dtype=torch.long, device=q.device)
    for j in range(STEPS):
        tokens[:, j] = torch.multinomial(q[j][prev], 1, generator=gen)[:, 0]
        prev = tokens[:, j]
    return tokens, prev


def simulate(
    chain: Chain, rule: str, n: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """One verification round per trial under ``rule`` (CPU, float64)."""
    gen = torch.Generator().manual_seed(seed)
    p, q = _tables(chain)
    drafts, _ = draw_blocks(p, q, n, gen)
    sampled = torch.full((n, STEPS + 1), -1, dtype=torch.long)
    count = torch.zeros(n, dtype=torch.long)
    alive = torch.ones(n, dtype=torch.bool)
    prev = torch.zeros(n, dtype=torch.long)
    for j in range(STEPS):
        d = drafts[:, j]
        p_row, q_row = p[j][prev], q[j][prev]
        ratio = p_row.gather(1, d[:, None])[:, 0] / q_row.gather(1, d[:, None])[:, 0]
        if rule == "always_accept":
            ratio = torch.full_like(ratio, 2.0)
        elif rule == "sqrt_ratio":
            ratio = ratio.sqrt()  # accepts too often for ratio < 1, too rarely above
        accept = torch.rand(n, generator=gen, dtype=torch.float64) < ratio.clamp(
            max=1.0
        )
        residual = (p_row - q_row).clamp(min=0)
        if rule == "resample_from_target":
            residual = p_row
        residual = residual + (residual.sum(1, keepdim=True) == 0) * p_row
        fix = torch.multinomial(residual, 1, generator=gen)[:, 0]
        if rule == "reuse_draft":
            fix = d
        token = torch.where(accept, d, fix)
        sampled[alive, j] = token[alive]
        count += alive.long()
        alive = alive & accept
        prev = d
    bonus = torch.multinomial(p[STEPS][prev], 1, generator=gen)[:, 0]
    sampled[alive, STEPS] = bonus[alive]
    count += alive.long()
    return sampled.numpy(), count.numpy()


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", [3, 11, 29])
def test_exact_prefix_formula_equals_full_enumeration(seed):
    chain = random_chain(seed)
    emitted = brute_force_emitted(chain)
    assert sum(emitted.values()) == 1  # the sample space is complete
    assert max(len(s) for s in emitted) == STEPS + 1
    for level in range(1, STEPS + 2):
        for prefix in itertools.product(range(VOCAB), repeat=level):
            enumerated = sum(
                (
                    m
                    for s, m in emitted.items()
                    if len(s) >= level and s[:level] == prefix
                ),
                Fraction(0),
            )
            assert enumerated == exact_prefix_probability(chain, prefix), prefix
    # the first token is exactly a target sample (the lossless property)
    for x in range(VOCAB):
        assert exact_prefix_probability(chain, (x,)) == chain.p[0][0][x]


def test_the_toy_chain_exercises_rejection_and_acceptance():
    chain = random_chain(3)
    emitted = brute_force_emitted(chain)
    lengths = [sum(m for s, m in emitted.items() if len(s) == n) for n in range(1, 6)]
    assert lengths[0] > Fraction(1, 10) and lengths[STEPS] > Fraction(1, 20)
    assert all(m > 0 for m in lengths[: STEPS + 1])


def test_gate_accepts_the_correct_rule_and_rejects_biased_rules():
    chain = random_chain(3)
    sampled, num = simulate(chain, "correct", NUM_TRIALS, seed=1)
    assert gate(sampled, num, chain) == []
    for rule in ("always_accept", "resample_from_target", "sqrt_ratio", "reuse_draft"):
        sampled, num = simulate(chain, rule, NUM_TRIALS, seed=2)
        assert gate(sampled, num, chain), f"the gate did not notice {rule!r}"


def _gpu() -> bool:
    return torch.cuda.is_available()


@pytest.mark.skipif(not _gpu(), reason="needs the HIP rejection sampler (GPU)")
@pytest.mark.parametrize("cache_dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("seed", [3, 11])
def test_production_rejection_sampler_matches_exact_probabilities(cache_dtype, seed):
    from vllm.v1.worker.gpu.spec_decode.draft_logits import DraftLogitsCache
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import rejection_sample

    device = torch.device("cuda")
    base = random_chain(seed)
    p64, q64 = _tables(base)
    # the draft samples from, and the cache stores, the logits in ``cache_dtype``; the
    # sampler then sees exactly ``softmax(cache)`` as q.  Rebuild the chain on that q.
    q_cache = q64.log().to(cache_dtype)
    q_eff = torch.softmax(q_cache.double(), dim=-1)
    chain = Chain(p64.numpy(), q_eff.numpy())
    p_eff, q_eff = _tables(chain)

    gen = torch.Generator(device=device).manual_seed(seed)
    p_g, q_g = p_eff.to(device), q_eff.to(device)
    drafts, _ = draw_blocks(p_g, q_g, NUM_TRIALS, gen)
    prev = torch.cat([torch.zeros_like(drafts[:, :1]), drafts], dim=1)  # anchor, d_0..
    draft_logits = torch.empty(
        NUM_TRIALS, STEPS, VOCAB, dtype=cache_dtype, device=device
    )
    target_logits = torch.empty(NUM_TRIALS, STEPS + 1, VOCAB, device=device)
    for j in range(STEPS):
        draft_logits[:, j] = q_cache.to(device)[j][prev[:, j]]
    for j in range(STEPS + 1):
        target_logits[:, j] = p64.to(device)[j][prev[:, j]].log().float()

    num_logits = NUM_TRIALS * (STEPS + 1)
    draft_sampled = torch.zeros(NUM_TRIALS, STEPS + 1, dtype=torch.int32, device=device)
    draft_sampled[:, 1:] = drafts
    cache = DraftLogitsCache(1, 1, 1, dtype=cache_dtype, fill=0.0, device=device)
    cache.values = draft_logits
    cache.row_mapping = None
    sampled, num_sampled = rejection_sample(
        target_logits=target_logits.reshape(num_logits, VOCAB).contiguous(),
        draft_logits_cache=cache,
        draft_sampled=draft_sampled.reshape(-1),
        cu_num_logits=torch.arange(NUM_TRIALS + 1, dtype=torch.int32, device=device)
        * (STEPS + 1),
        pos=torch.arange(num_logits, dtype=torch.int64, device=device),
        idx_mapping=torch.arange(NUM_TRIALS, dtype=torch.int32, device=device),
        expanded_idx_mapping=torch.arange(
            NUM_TRIALS, dtype=torch.int32, device=device
        ).repeat_interleave(STEPS + 1),
        expanded_local_pos=torch.arange(
            STEPS + 1, dtype=torch.int32, device=device
        ).repeat(NUM_TRIALS),
        temperature=torch.ones(NUM_TRIALS, dtype=torch.float32, device=device),
        seed=torch.arange(NUM_TRIALS, dtype=torch.int64, device=device) + 1000 * seed,
        num_speculative_steps=STEPS,
    )
    torch.accelerator.synchronize()
    sampled_np, num_np = sampled.cpu().numpy(), num_sampled.cpu().numpy()
    assert num_np.min() >= 1 and num_np.max() == STEPS + 1
    assert (num_np > 1).mean() > 0.1, "the test must exercise accepted drafts"
    failures = gate(sampled_np, num_np, chain)
    assert not failures, "\n".join(failures)
