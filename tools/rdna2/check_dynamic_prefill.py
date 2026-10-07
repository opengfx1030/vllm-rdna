#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Policy invariants for the adaptive prefill controller (VLLM_RDNA_DYNAMIC_PREFILL).

Drives the controller with synthetic windows -- no GPU, no engine -- and asserts the
rules that matter, so a change to the control law fails here instead of on the box:

  1. off by default: maybe_create() returns None, so the static path is untouched;
  2. alone: uncapped at interval 1 (the static lpt=0 solo behaviour);
  3. requests arriving one after another: the conservative cap at the interval ceiling;
  4. a batch (arrivals together): top cap at the interval ceiling;
  5. that shape is an episode property -- a quiet window cannot climb back above the
     conservative cap while the box is still shared, and nothing shortens the interval
     unless the fine ratchet is asked for;
  6. floor broken: jump the interval straight to the ceiling in one window;
  7. floor broken at the ceiling: recover by lowering the cap;
  8. slack and prefill work in flight: raise the cap, after dwell, and remember a cap
     that broke the floor so it is not retried;
  9. no prefill in the window: neither cap nor interval moves (nothing to speed up);
 10. the emitted values always stay inside the configured ladder / interval ceiling;
 11. the fine ratchet (VLLM_RDNA_DYN_IVL_FINE=1) gives the interval back a step at a
     time, only while the rolling floor minimum is well clear of the target.

What it does NOT validate: the control law against real hardware. The rates here are
made up, so this says "the policy reacts correctly to the numbers it is given", not
"these numbers are reachable". Validate that on the box with bench-prefill-itl.py.

    .venv/bin/python tools/rdna2/check_dynamic_prefill.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from vllm.logger import init_logger  # noqa: E402
from vllm.v1.core.sched.dynamic_prefill import maybe_create  # noqa: E402

LOG = init_logger("check_dynamic_prefill")
STEP_S = 0.05


def window(c, n_dec, tps_per_dec, backlog=0, prefill_tokens=200, windows=1, inflight=1,
           queued=0):
    """Feed `windows` full measurement windows at a fixed decoder population/rate.

    `inflight`/`queued` describe the batch shape; a window whose counts grew relative to
    the previous call is an arrival, and with `seq_gap_s` at 0 those read as sequential.
    """
    decoders = {f"r{i}" for i in range(n_dec)}
    steps = int(round(c.window_s / STEP_S))
    for _ in range(windows):
        acc = dict.fromkeys(decoders, 0.0)
        for _ in range(steps):
            c.begin_step(STEP_S)
            c.note_scheduled(decoders, prefill_tokens, backlog, inflight, queued)
            for req_id in decoders:
                acc[req_id] += tps_per_dec * STEP_S
                whole = int(acc[req_id])
                acc[req_id] -= whole
                c.note_tokens(req_id, whole)


def make(base_lpt=256, base_ivl=4, **env):
    for key, value in env.items():
        os.environ[f"VLLM_RDNA_DYN_{key}"] = str(value)
    os.environ["VLLM_RDNA_DYNAMIC_PREFILL"] = "1"
    return maybe_create(base_lpt, base_ivl, LOG)


def main() -> None:
    os.environ.pop("VLLM_RDNA_DYNAMIC_PREFILL", None)
    assert maybe_create(256, 4, LOG) is None, "must be inert without the env opt-in"
    assert maybe_create(256, 4, LOG) is None
    print("1/11 inert by default: ok")

    c = make(baseline_lpt := 256, 4)
    assert (c.lpt, c.ivl) == (256, 4.0), f"baseline not honoured: {c.lpt} {c.ivl}"
    window(c, n_dec=0, tps_per_dec=0)
    assert (c.lpt, c.ivl) == (0, 1.0), f"solo cell missed: {c.lpt} {c.ivl}"
    print(f"2/11 alone -> cap {c.lpt} (uncapped) interval {c.ivl}: ok")

    c_seq = make(256, 4)
    c_seq.seq_gap_s = 0.0  # real test steps are microseconds apart
    window(c_seq, n_dec=0, tps_per_dec=0)
    window(c_seq, n_dec=0, tps_per_dec=0, inflight=2)
    assert (c_seq.lpt, c_seq.ivl) == (512, 8.0), \
        f"sequential sharing must hold the conservative cell, got {c_seq.lpt} {c_seq.ivl}"
    print(f"3/11 sequential arrivals -> cap {c_seq.lpt} interval {c_seq.ivl}: ok")

    c_bat = make(256, 4)
    window(c_bat, n_dec=0, tps_per_dec=0)
    window(c_bat, n_dec=0, tps_per_dec=0, inflight=3)
    assert (c_bat.lpt, c_bat.ivl) == (1024, 8.0), \
        f"a batch keeps the top cap at the ceiling, got {c_bat.lpt} {c_bat.ivl}"
    print(f"4/11 batch -> cap {c_bat.lpt} interval {c_bat.ivl}: ok")

    for _ in range(4):  # plenty of slack, but the episode stays sequential
        window(c_seq, n_dec=3, tps_per_dec=20.0, inflight=2)
    assert c_seq.lpt == 512, f"climbed above the shared cap: {c_seq.lpt}"
    assert c_seq.ivl == 8.0, f"shortened with the fine ratchet off: {c_seq.ivl}"
    print(f"5/11 slack while shared-sequential -> still cap {c_seq.lpt} ivl {c_seq.ivl}: ok")

    c_br = make(1024, 4)
    window(c_br, n_dec=3, tps_per_dec=2.0, inflight=3)
    assert (c_br.lpt, c_br.ivl) == (1024, 8.0), \
        f"floor break must jump to the interval ceiling at once: {c_br.lpt} {c_br.ivl}"
    print(f"6/11 floor broken -> interval {c_br.ivl} in one window: ok")

    window(c_br, n_dec=3, tps_per_dec=2.0, inflight=3)
    assert (c_br.lpt, c_br.ivl) == (512, 8.0), \
        f"floor still broken at the ceiling: want a smaller cap, got {c_br.lpt} {c_br.ivl}"
    print(f"7/11 floor broken at ceiling -> cap lowered to {c_br.lpt}: ok")

    c3 = make(256, 4, WINDOW_S=1.0, LOG_EVERY=1e9)
    window(c3, n_dec=3, tps_per_dec=20.0, inflight=3, windows=3)  # reach lpt 1024
    while c3.lpt != 1024:
        window(c3, n_dec=3, tps_per_dec=20.0, inflight=3, windows=3)
    window(c3, n_dec=3, tps_per_dec=2.0, inflight=3, windows=2)  # 1024 breaks the floor
    assert (("2+", 0), 1024) in c3._blocked, f"failed cap not remembered: {c3._blocked}"
    assert c3.lpt == 512, f"failed cap must be given up: {c3.lpt}"
    assert c3._higher_rung(1024) is None, "retrying a cap that already broke the floor here"
    print(f"8/11 failed cap remembered: blocked={list(c3._blocked)}: ok")

    c5 = make(256, 4, WINDOW_S=1.0, LOG_EVERY=1e9)
    window(c5, n_dec=3, tps_per_dec=20.0, prefill_tokens=0, inflight=3, windows=4)
    assert (c5.lpt, c5.ivl) == (256, 4.0), \
        f"moved without prefill work in flight: {c5.lpt} {c5.ivl}"
    print(f"9/11 no prefill in flight -> held at lpt={c5.lpt} ivl={c5.ivl}: ok")

    c4 = make(256, 4, LOG_EVERY=1e9)
    for tps in (0.0, 2.0, 20.0, 2.0):
        window(c4, n_dec=(0 if tps == 0.0 else 3), tps_per_dec=tps, windows=3)
        assert c4.lpt in c4.ladder, f"cap {c4.lpt} outside ladder {c4.ladder}"
        assert 1.0 <= c4.ivl <= c4.ivl_max, f"interval {c4.ivl} outside bounds"
    print(f"10/11 values in range: lpt={c4.lpt} ladder={c4.ladder} ivl={c4.ivl}: ok")

    # The ratchet is the only path that shortens the interval, and it needs the wider
    # margin (headroom x 1.5 = 1.95x target here): a floor of 8 t/s is above the cap-raise
    # threshold but must not move anything.
    c_f = make(1024, 8, WINDOW_S=1.0, LOG_EVERY=1e9, IVL_FINE=1)
    window(c_f, n_dec=3, tps_per_dec=8.0, inflight=3, windows=3)
    assert c_f.ivl == 8.0, f"ratcheted without the extra margin: {c_f.ivl}"
    window(c_f, n_dec=3, tps_per_dec=20.0, inflight=3, windows=6)
    assert c_f.lpt == 1024, f"the ratchet must not touch the cap: {c_f.lpt}"
    assert 6.0 <= c_f.ivl < 8.0, \
        f"fine ratchet should give back a step at a time: ivl={c_f.ivl}"
    ratcheted = c_f.ivl
    window(c_f, n_dec=0, tps_per_dec=0, inflight=3)  # shared, no decoder at all
    assert c_f.ivl == ratcheted, \
        f"a no-decoder window undid the ratchet: {c_f.ivl} after {ratcheted}"
    c_no = make(1024, 5.0, WINDOW_S=1.0, LOG_EVERY=1e9)  # ratchet off
    window(c_no, n_dec=0, tps_per_dec=0, inflight=3)
    assert (c_no.lpt, c_no.ivl) == (1024, 8.0), \
        f"without the ratchet the shape cell owns the interval: {c_no.lpt} {c_no.ivl}"
    print(f"11/11 fine ratchet: 8.0 -> {c_f.ivl} in steps, cap {c_f.lpt} held, "
          f"no-decoder window kept it (off: {c_no.ivl}): ok")

    print("\nall policy invariants hold "
          f"(baseline {baseline_lpt} -> live lpt={c.lpt} ivl={c.ivl}, "
          f"learned={ {k: v[0] for k, v in c._table.items()} })")


if __name__ == "__main__":
    main()
