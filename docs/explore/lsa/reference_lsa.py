# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU reference for Latent Sparse Attention.

Clean-room math from Kubicek et al., arXiv:2610.42147. No file from
https://github.com/Prefrontal-AI/latent-sparse-attention is copied.

The Ward merge cost is the increase in within-cluster sum of squares
(Ward, 1963), implemented as

    (n_a * n_b) / (n_a + n_b) * ||mean_a - mean_b||^2

The paper's Algorithm 1 in the HTML is missing lines, so this is the
criterion Section 3.1 names, not their kernel. Merges are the mutual
cheapest legal pairs, which is the parallel round the text describes.
A pair whose combined key count would exceed ``n_b_max`` is skipped.

Centroids used for scoring are the L2 normalisation of the key sum,
as Section 3.1 states, not the mean. Distillation (Equation 2) is the
forward KL of the head-averaged attention against the index-space
softmax. Index vectors are L2-normalised before that dot, because
Section 3.1 normalises them before they are used. Equation 2 writes
the dot and does not repeat the word "normalise"; this is that reading.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


@dataclass
class Bucket:
    """One Ward bucket: key sum, key count, and token positions."""

    total: list[float]
    count: int
    positions: list[int] = field(default_factory=list)

    def centroid(self) -> list[float]:
        return l2_normalize(self.total)


@dataclass
class Node:
    """One tree node over ``arity`` children. Leaves are buckets."""

    total: list[float]
    children: list[Bucket | Node]

    def centroid(self) -> list[float]:
        return l2_normalize(self.total)


def l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0.0:
        return list(vector)
    return [x / norm for x in vector]


def dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def pool_keys(
    keys: list[list[float]],
    positions: list[int],
    n_pool: int,
) -> list[Bucket]:
    """Sum every ``n_pool`` consecutive keys into one pool bucket.

    Each bucket's ``count`` is ``n_pool`` original keys, matching the
    paper's initialisation ``|b| <- n_pool``.
    """
    if n_pool < 1:
        raise ValueError("n_pool must be positive")
    if len(keys) != len(positions):
        raise ValueError("keys and positions differ in length")
    if len(keys) % n_pool:
        raise ValueError("key count is not divisible by n_pool")
    width = len(keys[0]) if keys else 0
    pooled: list[Bucket] = []
    for start in range(0, len(keys), n_pool):
        total = [0.0] * width
        pos: list[int] = []
        for offset in range(n_pool):
            row = keys[start + offset]
            for dim, value in enumerate(row):
                total[dim] += value
            pos.append(positions[start + offset])
        pooled.append(Bucket(total, n_pool, pos))
    return pooled


def _ward_delta(left: Bucket, right: Bucket) -> float:
    scale = (left.count * right.count) / (left.count + right.count)
    dist = 0.0
    for a, b in zip(left.total, right.total):
        diff = a / left.count - b / right.count
        dist += diff * diff
    return scale * dist


def ward_cluster(
    buckets: list[Bucket],
    m: int,
    n_b_max: int,
) -> list[Bucket]:
    """Merge mutual cheapest pairs until ``m`` buckets remain.

    Stops early when every remaining pair would exceed ``n_b_max``.
    The returned list keeps unmerged buckets first, in their previous
    order, then the buckets created in that round.
    """
    if m < 1:
        raise ValueError("m must be positive")
    current = [
        Bucket(list(b.total), b.count, list(b.positions)) for b in buckets
    ]
    while len(current) > m:
        best = [-1] * len(current)
        best_cost = [math.inf] * len(current)
        for i, left in enumerate(current):
            for j, right in enumerate(current):
                if i == j:
                    continue
                if left.count + right.count > n_b_max:
                    continue
                cost = _ward_delta(left, right)
                better = cost < best_cost[i] or (
                    cost == best_cost[i] and (best[i] < 0 or j < best[i])
                )
                if better:
                    best_cost[i] = cost
                    best[i] = j
        pairs = [
            (i, j)
            for i, j in enumerate(best)
            if j > i and best[j] == i
        ]
        if not pairs:
            break
        consumed = {i for pair in pairs for i in pair}
        merged: list[Bucket] = []
        for i, j in pairs:
            left, right = current[i], current[j]
            total = [a + b for a, b in zip(left.total, right.total)]
            merged.append(
                Bucket(
                    total,
                    left.count + right.count,
                    left.positions + right.positions,
                )
            )
        current = [b for i, b in enumerate(current) if i not in consumed]
        current.extend(merged)
    return current


def _softmax(logits: list[float]) -> list[float]:
    peak = max(logits)
    exps = [math.exp(x - peak) for x in logits]
    total = sum(exps)
    return [x / total for x in exps]


def _kl(target: list[float], student: list[float]) -> float:
    total = 0.0
    for left, right in zip(target, student):
        if left > 0.0:
            total += left * math.log(left / max(right, 1e-12))
    return total


def expand_selection(
    query: list[float],
    buckets: list[Bucket],
    *,
    k: int,
    p: int,
    n_sink: int,
    n_window: int,
    query_pos: int,
) -> list[int]:
    """Flat top-``p`` over centroids, then sink + long-range + window.

    Scores are the dot of the unit query and the unit key-sum, which is
    Section 3.2. Expansion walks buckets best first and skips positions
    already covered by the sink or the window.
    """
    if p < 1 or k < 0:
        raise ValueError("p must be positive and k non-negative")
    unit_q = l2_normalize(query)
    scored = [(dot(unit_q, bucket.centroid()), bucket) for bucket in buckets]
    scored.sort(key=lambda item: (-item[0], min(item[1].positions, default=0)))
    chosen = [bucket for _, bucket in scored[:p]]
    sink = list(range(min(n_sink, query_pos)))
    window_start = max(0, query_pos - n_window)
    window = list(range(window_start, query_pos))
    covered = set(sink) | set(window)
    long_range: list[int] = []
    for bucket in chosen:
        ordered = sorted(bucket.positions)
        for pos in ordered:
            if pos < 0 or pos >= query_pos or pos in covered:
                continue
            long_range.append(pos)
            covered.add(pos)
            if len(long_range) == k:
                break
        if len(long_range) == k:
            break
    return sink + long_range + window


def build_tree(buckets: list[Bucket], arity: int) -> list[list[Bucket | Node]]:
    """Group adjacent nodes ``arity`` at a time until one parent remains.

    Level 0 is the buckets. Higher levels are ``Node`` values. This is
    the fixed grouping the tree search descends; it is not a second Ward
    pass. The paper clusters each level with Ward inside a block. The
    reference uses adjacent groups so a CPU test can name the parent.
    """
    if arity < 2:
        raise ValueError("arity must be at least 2")
    levels: list[list[Bucket | Node]] = [list(buckets)]
    while len(levels[-1]) > 1:
        previous = levels[-1]
        parents: list[Bucket | Node] = []
        for start in range(0, len(previous), arity):
            group = previous[start : start + arity]
            if len(group) == 1 and isinstance(group[0], Bucket):
                parents.append(group[0])
                continue
            width = len(group[0].total)
            total = [0.0] * width
            for child in group:
                for dim, value in enumerate(child.total):
                    total[dim] += value
            parents.append(Node(total, list(group)))
        if len(parents) == len(previous):
            break
        levels.append(parents)
    return levels


def tree_select(
    query: list[float],
    buckets: list[Bucket],
    *,
    arity: int,
    beam: int,
    k: int,
    p: int,
    n_sink: int,
    n_window: int,
    query_pos: int,
) -> list[int]:
    """Beam-search the adjacent tree, then the same expansion as flat."""
    if beam < 1:
        raise ValueError("beam must be positive")
    levels = build_tree(buckets, arity)
    unit_q = l2_normalize(query)
    current: list[Bucket | Node] = list(levels[-1])
    for _ in range(len(levels) - 1):
        scored = [(dot(unit_q, node.centroid()), node) for node in current]
        scored.sort(key=lambda item: -item[0])
        kept = [node for _, node in scored[:beam]]
        nxt: list[Bucket | Node] = []
        for node in kept:
            if isinstance(node, Node):
                nxt.extend(node.children)
            else:
                nxt.append(node)
        current = nxt
    leaves = [node for node in current if isinstance(node, Bucket)]
    return expand_selection(
        query,
        leaves,
        k=k,
        p=p,
        n_sink=n_sink,
        n_window=n_window,
        query_pos=query_pos,
    )


def softmax_layer_ids(layer_types: list[str]) -> list[int]:
    """Indices whose type is a softmax attention, not GDN."""
    softmax = {"full_attention", "qwen_sparse_attention", "indexed_attention"}
    return [i for i, name in enumerate(layer_types) if name in softmax]


def kv_bytes_per_token(
    n_layers: int,
    kv_heads_per_rank: int,
    head_dim: int,
    bytes_per_elem: int = 2,
) -> int:
    """fp16/bf16 K and V bytes for one token on one rank."""
    return n_layers * 2 * kv_heads_per_rank * head_dim * bytes_per_elem


def heads_per_rank(total_kv_heads: int, tp: int) -> int:
    """Match ``max(1, total // tp)`` in the Qwen attention modules."""
    return max(1, total_kv_heads // tp)


def centroid_bytes(
    n_tokens: int,
    d_idx: int = 64,
    keys_per_bucket: int = 8,
    bytes_per_elem: int = 2,
) -> int:
    """Bytes of one layer's unit centroids.

    Paper Appendix F: ``2 * d_idx * T / n_bar_b = 160 MiB`` at
    ``T = 10 * 2**20``, ``d_idx = 64``, ``n_bar_b = 8``, two-byte
    elements. This is that product.
    """
    if n_tokens % keys_per_bucket:
        raise ValueError("token count is not divisible by keys_per_bucket")
    buckets = n_tokens // keys_per_bucket
    return buckets * d_idx * bytes_per_elem


def _matvec(matrix: list[list[float]], vector: list[float]) -> list[float]:
    width = len(matrix[0])
    return [
        sum(matrix[row][col] * vector[row] for row in range(len(vector)))
        for col in range(width)
    ]


def _add_vec(left: list[float], right: list[float]) -> list[float]:
    return [a + b for a, b in zip(left, right)]


def toy_attention_kl(
    queries: list[list[list[float]]],
    keys: list[list[float]],
    w_q: list[list[float]],
    w_k: list[list[float]],
    tau: float,
) -> float:
    """Mean forward KL of Equation 2 on one causal toy layer.

    ``queries[i][h]`` is head ``h`` at position ``i``. ``keys[j]`` is the
    shared attention key. The target averages the head softmaxes. The
    student dots L2-normalised index vectors and divides by ``tau``.
    Positions with fewer than two legal keys are skipped.
    """
    if tau <= 0.0:
        raise ValueError("tau must be positive")
    n_tokens = len(keys)
    n_heads = len(queries[0])
    head_dim = len(keys[0])
    scale = 1.0 / math.sqrt(head_dim)
    total = 0.0
    counted = 0
    index_keys = [l2_normalize(_matvec(w_k, key)) for key in keys]
    for i in range(n_tokens):
        if i < 2:
            continue
        head_probs = []
        for h in range(n_heads):
            logits = [
                dot(queries[i][h], keys[j]) * scale for j in range(i)
            ]
            head_probs.append(_softmax(logits))
        target = [
            sum(head_probs[h][j] for h in range(n_heads)) / n_heads
            for j in range(i)
        ]
        summed = [0.0] * len(w_q[0])
        for h in range(n_heads):
            summed = _add_vec(summed, _matvec(w_q, queries[i][h]))
        index_q = l2_normalize(summed)
        student_logits = [
            dot(index_q, index_keys[j]) / tau for j in range(i)
        ]
        total += _kl(target, _softmax(student_logits))
        counted += 1
    if counted == 0:
        raise ValueError("sequence is too short to score a KL")
    return total / counted


def train_toy_indexer(
    steps: int = 40,
    lr: float = 0.35,
    seed: int = 1,
    eps: float = 1e-3,
) -> list[float]:
    """Finite-difference SGD on ``W_Q``, ``W_K`` and ``tau`` of a toy layer.

    The base model (queries and keys) stays frozen. Returns the KL after
    each step, including the initial value at index 0.
    """
    rng = random.Random(seed)
    n_tokens, n_heads, head_dim, d_idx = 6, 2, 4, 3

    def rand_vec() -> list[float]:
        return [rng.uniform(-0.8, 0.8) for _ in range(head_dim)]

    queries = [
        [rand_vec() for _ in range(n_heads)] for _ in range(n_tokens)
    ]
    keys = [rand_vec() for _ in range(n_tokens)]
    w_q = [
        [rng.uniform(-0.2, 0.2) for _ in range(d_idx)] for _ in range(head_dim)
    ]
    w_k = [
        [rng.uniform(-0.2, 0.2) for _ in range(d_idx)] for _ in range(head_dim)
    ]
    tau = 1.0

    def loss() -> float:
        return toy_attention_kl(queries, keys, w_q, w_k, tau)

    history = [loss()]
    params: list[tuple[list[list[float]], int, int]] = []
    for matrix in (w_q, w_k):
        for row in range(len(matrix)):
            for col in range(len(matrix[0])):
                params.append((matrix, row, col))
    for _ in range(steps):
        base = history[-1]
        grads: list[float] = []
        for matrix, row, col in params:
            old = matrix[row][col]
            matrix[row][col] = old + eps
            grads.append((loss() - base) / eps)
            matrix[row][col] = old
        tau_up = tau + eps
        tau_grad = (
            toy_attention_kl(queries, keys, w_q, w_k, tau_up) - base
        ) / eps
        for (matrix, row, col), grad in zip(params, grads):
            matrix[row][col] -= lr * grad
        tau = max(tau - lr * tau_grad, 0.05)
        history.append(loss())
    return history


def _demo() -> None:
    keys = [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]]
    pooled = pool_keys(keys, [0, 1, 2, 3], 1)
    clustered = ward_cluster(pooled, m=2, n_b_max=2)
    print("ward buckets", [b.positions for b in clustered])
    history = train_toy_indexer()
    print(f"toy KL {history[0]:.4f} -> {history[-1]:.4f}")


if __name__ == "__main__":
    _demo()
