#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""fp64 oracle for ``gdn_prefill_delta_h_rdna2`` (HIP) and the Triton reference.

The pairwise parity test (HIP against ``chunk_gated_delta_rule_fwd_h``) compares
two implementations that both round h, v_new and the gated v to fp16, so one
cancelling element of ``v_new = u - w.h`` can miss its rtol/atol without telling
which side is off. This tool runs both on the same inputs and measures each
against the exact fp64 recurrence (``_oracle_fp64`` of the test module).

    # the TP1 outlier of G2 (H=48, Hg=16, T=640, no initial state) + its detail
    HIP_VISIBLE_DEVICES=0 python tools/rdna/rdna3/gdn_delta_h_oracle.py \
        --cases failing --dump w7800-case.pt
    # the whole test grid
    HIP_VISIBLE_DEVICES=0 python tools/rdna/rdna3/gdn_delta_h_oracle.py --cases sweep
    # the same inputs on another arch (a V620): the HIP kernel's bits against
    # the dump of the first device, and both devices against the oracle
    HIP_VISIBLE_DEVICES=<v620> python tools/rdna/rdna3/gdn_delta_h_oracle.py \
        --replay w7800-case.pt

Output order of every row is (h, v_new, final_state). ``bad`` counts elements
outside rtol = atol = 1e-3 (the tolerance of the pairwise test).
"""

import argparse
import importlib.util
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
TEST = REPO / "tests/kernels/quantization/test_gdn_prefill_delta_h_rdna2.py"
NAMES = ("h", "v_new", "final")
RTOL = ATOL = 1e-3


def load_helpers():
    spec = importlib.util.spec_from_file_location("delta_h_helpers", TEST)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cases(which: str):
    if which == "failing":
        return [(48, 16, [640], False)]
    return [
        (h, hg, seq, init)
        for h, hg in [(12, 4), (4, 4), (24, 8), (48, 16)]
        for seq in ([64], [192], [640], [100], [137], [63])
        for init in (False, True)
    ]


def bad(x: torch.Tensor, y: torch.Tensor) -> int:
    close = torch.isclose(x.double().cpu(), y.double().cpu(), rtol=RTOL, atol=ATOL)
    return int((~close).sum())


def stats(x: torch.Tensor, exact: torch.Tensor):
    diff = x.double().cpu() - exact
    return float(diff.abs().max()), float(diff.pow(2).mean().sqrt()), bad(x, exact)


def ratio(hip: float, triton: float) -> str:
    """HIP over Triton error; both zero (exact against the oracle) reads as 1."""
    if triton == 0.0:
        return "1.00" if hip == 0.0 else "inf"
    return f"{hip / triton:.2f}"


def chunk_of(cu: list[int], token: int, block: int) -> int:
    """Flat chunk index of a packed token (chunks of all sequences in order)."""
    index = 0
    for begin, end in zip(cu[:-1], cu[1:]):
        if token < end:
            return index + (token - begin) // block
        index += (end - begin + block - 1) // block
    raise ValueError(token)


def element_detail(mod, inputs, hip, ref, exact):
    """The first v_new elements outside the pairwise tolerance, in detail."""
    k, w, u, g, cu_seqlens, _ = inputs
    hip_v, ref_v = hip[1].double().cpu(), ref[1].double().cpu()
    outside = (~torch.isclose(hip_v, ref_v, rtol=RTOL, atol=ATOL)).nonzero()
    cu = cu_seqlens.tolist()
    for b, t, head, v in outside[:4].tolist():
        c = chunk_of(cu, t, mod.BT)
        h_c = exact[0][0, c, head]  # [V, K], exact state at the chunk start
        mag = float(w[0, t, head].double().cpu().abs() @ h_c[v].abs())
        print(
            f"  v_new[{b},{t},{head},{v}] chunk {c}: hip {hip_v[b, t, head, v]:.6f}"
            f" ref {ref_v[b, t, head, v]:.6f} exact {exact[1][b, t, head, v]:.6f}"
            f" |hip-exact| {abs(hip_v[b, t, head, v] - exact[1][b, t, head, v]):.3e}"
            f" |ref-exact| {abs(ref_v[b, t, head, v] - exact[1][b, t, head, v]):.3e}"
            f" u {float(u[b, t, head, v]):.4f} sum|w||h| {mag:.4f}"
        )


def run_case(mod, case, dump):
    h, hg, seq, init = case
    torch.manual_seed(0)
    inputs = mod._make_inputs(seq, h, hg, seed=42, with_initial_state=init)
    ref = mod._run_ref(*inputs, True)
    hip = mod._run_hip(*inputs, True)
    exact = mod._oracle_fp64(*inputs[:4], inputs[4], inputs[5])
    tag = f"H{h}/Hg{hg} T={sum(seq) if len(seq) == 1 else seq} init={int(init)}"
    flagged = 0
    worse = 0
    for name, x_hip, x_ref, x_exact in zip(NAMES, hip, ref, exact):
        mh, rh, bh = stats(x_hip, x_exact)
        mr, rr, br = stats(x_ref, x_exact)
        pair = bad(x_hip, x_ref)
        flagged += pair > 0
        worse += rh > 1.25 * rr
        print(
            f"{tag:26} {name:6} pair_bad {pair:3d} | hip: max {mh:.3e}"
            f" rms {rh:.3e} bad {bh:3d} | triton: max {mr:.3e}"
            f" rms {rr:.3e} bad {br:3d} | rms hip/triton {ratio(rh, rr)}"
        )
    if flagged:
        element_detail(mod, inputs, hip, ref, exact)
    if dump:
        torch.save(
            {
                "arch": torch.cuda.get_device_properties(0).gcnArchName,
                "case": case,
                "inputs": [None if x is None else x.cpu() for x in inputs],
                "hip": [x.cpu() for x in hip],
                "ref": [x.cpu() for x in ref],
            },
            dump,
        )
        print(f"dumped {dump}")
    return flagged, worse


def bit_diff(x: torch.Tensor, y: torch.Tensor):
    xb = (
        x.cpu().contiguous().view(torch.int16 if x.element_size() == 2 else torch.int32)
    )
    yb = y.cpu().contiguous().view(xb.dtype)
    differ = xb != yb
    maxdiff = float((x.double().cpu() - y.double().cpu()).abs().max())
    return int(differ.sum()), maxdiff


def replay(mod, path):
    saved = torch.load(path)
    arch = torch.cuda.get_device_properties(0).gcnArchName
    inputs = [None if x is None else x.cuda() for x in saved["inputs"]]
    print(f"dump from {saved['arch']}, this device {arch}, case {saved['case']}")
    ref = mod._run_ref(*inputs, True)
    hip = mod._run_hip(*inputs, True)
    exact = mod._oracle_fp64(*inputs[:4], inputs[4], inputs[5])
    for name, a, b, c, d, e in zip(NAMES, hip, saved["hip"], ref, saved["ref"], exact):
        n_hip, m_hip = bit_diff(a, b)
        n_ref, m_ref = bit_diff(c, d)
        print(
            f"{name:6} HIP here vs HIP dump: {n_hip} elements differ bitwise"
            f" (max {m_hip:.3e}); Triton here vs Triton dump: {n_ref} (max {m_ref:.3e})"
        )
        for label, x in (
            ("hip here", a),
            ("hip dump", b),
            ("triton here", c),
            ("triton dump", d),
        ):
            mx, rms, nbad = stats(x, e)
            print(f"       vs fp64, {label:12}: max {mx:.3e} rms {rms:.3e} bad {nbad}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cases", choices=("failing", "sweep"), default="failing")
    parser.add_argument("--dump", help="save inputs and outputs of the first case")
    parser.add_argument("--replay", help="run the inputs of a dump on this device")
    args = parser.parse_args()
    mod = load_helpers()
    if args.replay:
        replay(mod, args.replay)
        return 0
    flagged = worse = 0
    for index, case in enumerate(cases(args.cases)):
        f, w = run_case(mod, case, args.dump if index == 0 else None)
        flagged += f
        worse += w
    print(
        f"{flagged} (case, output) rows outside the pairwise tolerance; "
        f"{worse} rows where the HIP rms error vs fp64 exceeds 1.25x the Triton one"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
