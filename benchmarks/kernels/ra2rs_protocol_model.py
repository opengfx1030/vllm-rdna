"""CPU model of the rdna_ar2_rs reduce-scatter + allgather protocol.

Mirrors the exact layout in csrc/rocm/rdna_ar2_rs.cuh: per-peer staging split into a
contrib region [sender][chunk] and a result region [chunk], with per-(sender, chunk,
phase) flags compared against a per-call generation. This is the gate that catches
index and ordering errors before a compile cycle; it found three real bugs in the
first draft of this design (staging layout, scatter loop coverage, unstaged own slice).
"""


def model(W, CH, rank_data):
    n = len(rank_data[0])
    assert n % CH == 0, "chunk_elems must divide n"
    ce = n // CH
    gen = 1

    contrib = [[[[0.0] * ce for _ in range(CH)] for _ in range(W)] for _ in range(W)]
    result = [[[0.0] * ce for _ in range(CH)] for _ in range(W)]
    flag_a = [[[0] * CH for _ in range(W)] for _ in range(W)]
    flag_b = [[[0] * CH for _ in range(W)] for _ in range(W)]
    fails = []

    for r in range(W):
        for c in range(CH):
            owner = c % W
            contrib[owner][r][c] = list(rank_data[r][c * ce:(c + 1) * ce])
            if r != owner:
                flag_a[owner][r][c] = gen

    for r in range(W):
        for c in range(CH):
            owner = c % W
            if owner != r:
                continue
            for j in range(W):
                if j == r:
                    continue
                if flag_a[r][j][c] < gen:
                    fails.append(f"owner{r} chunk{c}: no phase-A flag from {j}")
                    return None, fails
            acc = [0.0] * ce
            for j in range(W):
                for i in range(ce):
                    acc[i] += contrib[r][j][c][i]
            result[r][c] = acc
            for j in range(W):
                if j == r:
                    continue
                result[j][c] = list(acc)
                flag_b[j][r][c] = gen

    out = [[0.0] * n for _ in range(W)]
    for r in range(W):
        for c in range(CH):
            owner = c % W
            if owner != r and flag_b[r][owner][c] < gen:
                fails.append(f"rank{r} chunk{c}: no phase-B flag from owner{owner}")
                return None, fails
            out[r][c * ce:(c + 1) * ce] = result[r][c]
    return out, fails
