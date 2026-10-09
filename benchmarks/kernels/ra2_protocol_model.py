"""CPU model of the rdna_ar2 reduce-scatter + allgather protocol.

Simulates the flag handshake for arbitrary (W, CH) exactly as ra2_reduce_scatter /
ra2_all_gather do, then checks the numeric result equals a plain all-reduce. This is
the check that catches index/ordering errors before any kernel is written; the
previous two-shot attempt's index math was likewise verified correct here, which is
how its failure was localized to synchronization.
"""


def model(W, CH, rank_data):
    n = len(rank_data[0])
    assert n % CH == 0, "chunk_elems must divide n for this model"
    ce = n // CH

    stage = [[[[0.0] * ce for _ in range(CH)] for _ in range(W)] for _ in range(W)]
    flags_A = [[0] * CH for _ in range(W)]
    flags_B = [[0] * CH for _ in range(W)]
    gen = 1

    part = [[0.0] * ce for _ in range(W)]
    failures = []

    for r in range(W):
        for c in range(CH):
            owner = c % W
            src_slice = rank_data[r][c * ce:(c + 1) * ce]
            stage[owner][r][c] = list(src_slice)
            if owner != r:
                flags_A[owner][c] += 1

    for r in range(W):
        for c in range(CH):
            owner = c % W
            if owner != r:
                continue
            for j in range(W):
                if j == r:
                    continue
                if flags_A[r][c] < W - 1:
                    failures.append(f"rank{r} chunk{c}: only {flags_A[r][c]}/{W-1} contributions")
                    return None, failures
            acc = [0.0] * ce
            for j in range(W):
                for i in range(ce):
                    acc[i] += stage[r][j][c][i]
            stage[r][r][c] = list(acc)
            for j in range(W):
                if j == r:
                    continue
                stage[j][r][c] = list(acc)
            for j in range(W):
                if j != r:
                    flags_B[j][c] = gen

    out = [[0.0] * n for _ in range(W)]
    for r in range(W):
        for c in range(CH):
            owner = c % W
            if owner == r:
                out[r][c * ce:(c + 1) * ce] = stage[r][r][c]
                continue
            if flags_B[r][c] < gen:
                failures.append(f"rank{r} chunk{c}: missing flag_B from owner{owner}")
                return None, failures
            out[r][c * ce:(c + 1) * ce] = stage[r][owner][c]
    return out, failures

TESTS = [(2, 1), (2, 4), (4, 1), (4, 2), (4, 4), (4, 8), (8, 8), (8, 16)]
N = 32
allok = True
for W, CH in TESTS:
    data = [[float(r * 1000 + i) for i in range(N)] for r in range(W)]
    expect = [sum(data[r][i] for r in range(W)) for i in range(N)]
    out, fails = model(W, CH, data)
    if out is None:
        print(f"W={W:>2} CH={CH:>2}: FAIL {fails}")
        allok = False
        continue
    bad = [r for r in range(W) if out[r] != expect]
    tag = "ok" if not bad else f"MISMATCH ranks {bad}"
    print(f"W={W:>2} CH={CH:>2}: {tag}   traffic/rank={(W - 1) / W * 2:.3f}*N")
    if bad:
        allok = False

print()
print("balanced chunk ownership requires CH % W == 0 and CH >= W:")
for W in (2, 4, 8):
    good = [c for c in range(1, 17) if c % W == 0 and c >= W]
    print(f"  W={W}: valid CH = {good[:6]}")
print()
print("ALL PASS" if allok else "FAILURES PRESENT")
