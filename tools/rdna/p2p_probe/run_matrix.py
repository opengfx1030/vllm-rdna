#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Drive p2p_probe over the par1-cs25 V620 set (CPU side: no torch).

Usage: run_matrix.py <out.log> <hip ids csv> [suite ...]
Suites: info lat bw1 bwbi conc inter a2a
HIP ids are resolved to PCI buses first; the run refuses anything that is not
a gfx1030 V620 on the expected buses. Logical id = index into the csv.
"""

import itertools
import os
import subprocess
import sys

BIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "p2p_probe")
SLOT_A = {"ce", "d3", "d8", "db"}  # root c0
SLOT_B = {"0a", "0f", "14", "17"}  # root 00
SIZES = [4096, 16384, 65536, 1048576, 5242880, 20971520, 67108864]


def run(hip, stdin=None, args=()):
    env = dict(os.environ, HIP_VISIBLE_DEVICES=hip)
    r = subprocess.run(
        [BIN, *args], input=stdin, capture_output=True, text=True, env=env, timeout=3000
    )
    if r.returncode:
        raise SystemExit(f"probe failed rc={r.returncode}\n{r.stderr}")
    return r.stdout


def main():
    out, hip = sys.argv[1], sys.argv[2]
    suites = sys.argv[3:] or ["info", "lat", "bw1", "bwbi", "conc", "inter", "a2a"]
    info = run(hip, args=["info"])
    bus = {}
    for ln in info.splitlines():
        if ln.startswith("DEV"):
            p = ln.split()
            assert "gfx1030" in ln, ln
            bus[int(p[1])] = p[-1].split("=")[1].split(":")[1].lower()
    for d, b in bus.items():
        assert b in SLOT_A | SLOT_B, f"logical {d} is bus {b}, not a V620 of slot A/B"
    n = len(bus)
    A = [d for d in range(n) if bus[d] in SLOT_A]
    B = [d for d in range(n) if bus[d] in SLOT_B]
    lines = [f"# hip={hip} buses={bus} A={A} B={B}"]
    pairs = list(itertools.combinations(range(n), 2))

    def bw(size, xs, dur=0.4, blocks=128):
        lines.append(f"bw {dur} {size} {blocks} " + " ".join(xs))

    if "lat" in suites:
        for payload in (0, 4096, 16384, 65536):
            lines.append(
                f"lat 2000 {payload} " + " ".join(f"{a}:{b}" for a, b in pairs)
            )
    only = os.environ.get("BW1_ONLY")  # restrict bw1 to pairs touching this GPU
    if "bw1" in suites:  # one ordered pair at a time, push and pull
        for size in SIZES if only is None else (65536, 1048576, 5242880, 67108864):
            for a, b in itertools.permutations(range(n), 2):
                if only is not None and int(only) not in (a, b):
                    continue
                for m in "pl":
                    bw(size, [f"{m}:{a}>{b}"])
        for size in (65536, 1048576, 5242880, 67108864) if only is None else ():
            for a, b in itertools.permutations(range(n), 2):  # sdma
                bw(size, [f"s:{a}>{b}"])
    if "bwbi" in suites:  # both directions at once
        for size in (65536, 1048576, 5242880, 67108864):
            for a, b in pairs:
                bw(size, [f"p:{a}>{b}", f"p:{b}>{a}"])
    if "conc" in suites:
        k = min(len(A), len(B))
        for size in (65536, 1048576, 5242880, 20971520):
            for kk in (1, 2, 3, 4):
                if kk > k:
                    continue
                ab = [f"p:{A[i]}>{B[i]}" for i in range(kk)]
                ba = [f"p:{B[i]}>{A[i]}" for i in range(kk)]
                bw(size, ab)  # A -> B, kk pairs
                bw(size, ba)  # B -> A
                bw(size, ab + ba)  # bidirectional, kk pairs
            # intra-switch, all 4 GPUs of a switch at once, ring and all-to-all
            for grp in (A, B):
                if len(grp) < 2:
                    continue
                ring = [
                    f"p:{grp[i]}>{grp[(i + 1) % len(grp)]}" for i in range(len(grp))
                ]
                bw(size, ring)
                bw(size, [f"p:{a}>{b}" for a in grp for b in grp if a != b])
            if len(A) >= 2 and len(B) >= 2:
                r = lambda g: [f"p:{g[i]}>{g[(i + 1) % len(g)]}" for i in range(len(g))]
                bw(size, r(A) + r(B))  # both switches' rings together
            # fan-in / fan-out across the root link
            bw(size, [f"p:{a}>{B[0]}" for a in A])
            bw(size, [f"p:{A[0]}>{b}" for b in B])
    if "inter" in suites:  # does intra-switch traffic slow cross-root traffic
        # Disjoint endpoints: fg uses the first kk GPUs of A and B; bg rings run
        # only among the remaining GPUs of each switch.
        for size in (65536, 1048576, 5242880, 20971520):
            for kk in (1, 2):
                fgx = [f"p:{A[i]}>{B[i]}" for i in range(kk)]
                bw(size, fgx)
                bgx = []
                for g in (A[kk:], B[kk:]):
                    if len(g) == 2:
                        bgx += [f"bg:p:{g[0]}>{g[1]}", f"bg:p:{g[1]}>{g[0]}"]
                    elif len(g) > 2:
                        bgx += [
                            f"bg:p:{g[i]}>{g[(i + 1) % len(g)]}" for i in range(len(g))
                        ]
                bw(size, fgx + bgx)
                bw(size, bgx[:0] + [x[3:] for x in bgx])  # bg alone
    if "xr" in suites:  # cross-root flow-count / placement / block-count scan
        size = 1048576
        for kk in (2, 3, 4):
            for src in itertools.combinations(A, kk):
                for dst in itertools.combinations(B, kk):
                    bw(size, [f"p:{a}>{b}" for a, b in zip(src, dst)], dur=0.3)
        for blk in (8, 16, 32, 64, 128, 256, 512):
            kk = min(3, len(A), len(B))
            bw(size, [f"p:{A[i]}>{B[i]}" for i in range(kk)], dur=0.3, blocks=blk)
            bw(size, [f"p:{A[0]}>{B[0]}"], dur=0.3, blocks=blk)
    if "a2a" in suites:  # flat one-shot style all-to-all over every GPU
        for size in (4096, 16384, 65536, 1048576):
            bw(
                size,
                [f"p:{a}>{b}" for a in range(n) for b in range(n) if a != b],
                dur=0.6,
            )
            # hierarchical: all-to-all inside each switch, then 1 pair exchange
            if len(A) > 1 and len(B) > 1:
                intra = [
                    f"p:{g[i]}>{g[j]}"
                    for g in (A, B)
                    for i in range(len(g))
                    for j in range(len(g))
                    if i != j
                ]
                bw(size, intra, dur=0.6)
                bw(size, [f"p:{A[0]}>{B[0]}", f"p:{B[0]}>{A[0]}"], dur=0.6)
    if "hier" in suites:  # latency regime for the hierarchical all-reduce design
        for payload in (4096, 8192, 16384, 32768, 65536):
            for _rep in range(3):
                for kk in range(1, min(len(A), len(B)) + 1):
                    cross = [f"{A[i]}:{B[i]}" for i in range(kk)]
                    lines.append(
                        f"latc 2000 {payload} " + " ".join(cross)
                    )  # push+flag, one way, kk pairs at once
                    lines.append(
                        f"a2a 2000 {payload} "
                        + " ".join(f"{A[i]},{B[i]}" for i in range(kk))
                    )  # bidirectional exchange
                lines.append(
                    f"a2a 2000 {payload} "
                    + ",".join(map(str, A))
                    + " "
                    + ",".join(map(str, B))
                )  # intra a2a, both switches
                lines.append(
                    f"a2a 2000 {payload} " + ",".join(map(str, range(n)))
                )  # flat one-shot over every GPU
    if "sub" in suites:  # compact before/after subset, 3 repeats for spread
        for _rep in range(3):
            lines.append(f"# rep {_rep}")
            for size in (65536, 1048576):
                bw(size, [f"p:{A[0]}>{B[0]}"], dur=0.5)  # cross write
                bw(size, [f"l:{A[0]}>{B[0]}"], dur=0.5)  # cross read (pull)
                bw(size, [f"p:{A[0]}>{A[1]}"], dur=0.5)  # intra write
                bw(size, [f"l:{A[0]}>{A[1]}"], dur=0.5)  # intra read
                bw(size, [f"s:{A[0]}>{B[0]}"], dur=0.5)  # hipMemcpyPeerAsync cross
            for kk in (1, 2, 3, 4):
                bw(1048576, [f"p:{A[i]}>{B[i]}" for i in range(kk)], dur=0.5)
            bw(
                1048576,
                [f"p:{A[i]}>{B[i]}" for i in range(4)]
                + [f"p:{B[i]}>{A[i]}" for i in range(4)],
                dur=0.5,
            )
            bw(
                1048576, [f"p:{A[i]}>{A[(i + 1) % 4]}" for i in range(4)], dur=0.5
            )  # intra ring
            for payload in (4096, 16384, 65536):
                lines.append(
                    f"latc 2000 {payload} "
                    + " ".join(f"{A[i]}:{B[i]}" for i in range(4))
                )
                lines.append(f"a2a 2000 {payload} " + ",".join(map(str, range(n))))
    text = run(hip, stdin="\n".join(lines) + "\n", args=["batch"])
    with open(out, "w") as f:
        f.write(info + "\n".join(lines[:1]) + "\n" + text)
    print(f"wrote {out} ({len(lines)} commands)")


if __name__ == "__main__":
    main()
