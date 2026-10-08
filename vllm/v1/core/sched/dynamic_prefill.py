# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Adaptive prefill scheduling: live tuning of the chunk cap and the step cadence.

DISABLED BY DEFAULT. Opt in with ``VLLM_RDNA_DYNAMIC_PREFILL=1``; with any other value
the scheduler keeps its static configuration (every hook below is skipped).

The objective is deliberately *not* a target rate: maximise prefill throughput subject
to keeping every decoder's own rate at or above a floor. The measured grid (docs/rdna2)
gives the two knobs different roles, so the loop follows the *shape* of the batch rather
than searching the space:

* **alone** (at most one request in flight, nothing queued): uncapped at interval 1 --
  the static lpt=0 behaviour, the fastest prefill there is (1856 t/s / 18.7 s TTFT on
  a 34.7k prompt vs 1713 / 20.3 at cap 1024). Nothing can be starved, so this is also
  the only shape where a short interval buys real prefill;
* **batch** (several requests arriving together): their prefills end together, so no
  decoder ever sees one; top cap at the interval ceiling costs almost no throughput
  (1858 vs 1877 t/s measured at 3-way concurrency) and holds the floor;
* **sequential** (requests arriving one after another): the earlier ones start decoding
  while the later ones still prefill, and a 1024-token release step starves that decoder
  at *every* interval (measured floor 4.4 t/s at 1024+8 vs 7.0 at 512+8), so hold the
  conservative cap at the interval ceiling;
* a **measured floor below target** overrides all of it: straight to the ceiling
  interval, then down the cap ladder. A cap that failed is remembered per shape bucket,
  so the loop cannot oscillate back into the cell that broke the floor.
* an optional **fine ratchet** (`VLLM_RDNA_DYN_IVL_FINE=1`) gives cadence back once the
  floor has room: while prefill work is in flight and the *rolling* floor minimum is at
  least `headroom * 1.5` over target, the interval shrinks by `IVL_STEP` per dwell
  window. Off by default: throughput is flat in the interval, so this only buys TTFT in
  light mixed shapes, and the client judges the floor over 5 s windows while the tuner
  measures 2 s ones (measured bias ~1.9x, which is what the extra 1.5x margin covers).

A cell is only given up on measurement, never predicted: the shape branches above only
decide *which* cell to sit in, and the shape is what can actually be observed.

Env (read once, at construction):
  VLLM_RDNA_DYNAMIC_PREFILL  0      enable; anything but "1" leaves the static path
  VLLM_RDNA_DYN_FLOOR_TPS    5.0    per-decoder floor the cadence must not break
  VLLM_RDNA_DYN_WINDOW_S     2.0    measurement/decision window
  VLLM_RDNA_DYN_IVL_MAX      8.0    ceiling for the cadence interval
  VLLM_RDNA_DYN_LPT_MIN      256    chunk-cap ladder lower bound
  VLLM_RDNA_DYN_LPT_MAX      1024   ladder upper bound; 0 appends "uncapped" as top rung
  VLLM_RDNA_DYN_LPT_SOLO     0      cap while alone; 0 = uncapped, i.e. the biggest bite
  VLLM_RDNA_DYN_LPT_SHARED   512    cap held while requests arrive one after another
  VLLM_RDNA_DYN_HEADROOM     1.3    slack over the floor required before raising the cap
  VLLM_RDNA_DYN_DWELL        2      windows to hold after a change
  VLLM_RDNA_DYN_LOG_EVERY    15.0   seconds between LEARN summaries
  VLLM_RDNA_DYN_START_LPT    -      override the baseline cap (default: configured)
  VLLM_RDNA_DYN_START_IVL    -      override the baseline interval (default: configured)
  VLLM_RDNA_DYN_IVL_FINE     0      fine interval ratchet: shorten while floor has room
  VLLM_RDNA_DYN_IVL_STEP     0.5    interval each ratchet step gives back
"""

from __future__ import annotations

import logging
import os
import time
from collections import deque

RUNGS = (256, 512, 1024)  # chunk-cap rungs; 0 (uncapped) can be added as the top rung
_MIN_SAMPLE_S = 0.3  # a decoder needs this much active time to be judged in a window
_MAX_STEP_S = 10.0  # clamp a wedged step so one stall cannot poison a whole window
_SEQ_GAP_S = 2.0  # arrivals further apart than this are sequential, not a batch


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


class DynamicPrefillController:
    """Live tuner for (long_prefill_token_threshold, prefill_schedule_interval)."""

    def __init__(self, base_lpt: int, base_ivl: int, logger: logging.Logger):
        self.log = logger
        self.floor_tps = _env_float("VLLM_RDNA_DYN_FLOOR_TPS", 5.0)
        self.window_s = max(0.5, _env_float("VLLM_RDNA_DYN_WINDOW_S", 2.0))
        self.ivl_max = max(1.0, _env_float("VLLM_RDNA_DYN_IVL_MAX", 8.0))
        self.headroom = max(1.0, _env_float("VLLM_RDNA_DYN_HEADROOM", 1.3))
        self.dwell = max(1, _env_int("VLLM_RDNA_DYN_DWELL", 2))
        self.log_every = max(1.0, _env_float("VLLM_RDNA_DYN_LOG_EVERY", 15.0))
        self.fine_ivl = _env_int("VLLM_RDNA_DYN_IVL_FINE", 0) == 1
        self.ivl_step = max(0.1, _env_float("VLLM_RDNA_DYN_IVL_STEP", 0.5))
        # The ratchet needs more margin than a cap raise: the client judges the floor
        # over 5 s windows while this measures 2 s ones, and the measured bias is ~1.9x.
        self._ivl_margin = self.headroom * 1.5

        lpt_min = _env_int("VLLM_RDNA_DYN_LPT_MIN", 256)
        lpt_max = _env_int("VLLM_RDNA_DYN_LPT_MAX", 1024)
        if lpt_max == 0:  # 0 = uncapped, i.e. the largest bite, so it is the top rung
            ladder = [v for v in RUNGS if v >= lpt_min] + [0]
        else:
            ladder = [v for v in RUNGS if lpt_min <= v <= lpt_max]
        self.ladder = ladder or [lpt_max]

        # 0 (uncapped) is the solo cell and always ranks last, i.e. biggest, so the
        # ladder walk can treat it and the bounded rungs uniformly.
        solo = _env_int("VLLM_RDNA_DYN_LPT_SOLO", 0)
        self.solo_lpt = (
            min(self.ladder, key=lambda v: (abs(v - solo), v)) if solo else 0
        )
        if self.solo_lpt not in self.ladder:
            self.ladder = [*self.ladder, self.solo_lpt]
        bounded = [v for v in self.ladder if v > 0]
        self.batch_lpt = max(bounded)
        shared = _env_int("VLLM_RDNA_DYN_LPT_SHARED", 512)
        self.safe_lpt = min(bounded, key=lambda v: (abs(v - shared), v))

        start_lpt = _env_int("VLLM_RDNA_DYN_START_LPT", base_lpt)
        start_ivl = _env_float("VLLM_RDNA_DYN_START_IVL", base_ivl)
        self.lpt = min(self.ladder, key=lambda v: (abs(v - start_lpt), v))
        self.ivl = min(max(1.0, start_ivl), self.ivl_max)

        self._phase = 0.0  # cadence phase accumulator (fractional intervals)
        self._dt = 0.0
        self._active: dict[str, float] = {}  # req id -> seconds decoding this window
        self._tokens: dict[str, int] = {}  # req id -> accepted tokens this window
        self._win_prefill = 0
        self._win_elapsed = 0.0
        self._win_dec_peak = 0
        self._win_steps = 0
        self._win_alone = True
        self._win_sequential = False
        self._shared_sequential = False  # episode shape: conservative until alone again
        self._shared = False  # a sharing episode is in progress
        self.seq_gap_s = _SEQ_GAP_S
        self._pending = 0
        self._last_arrival: float | None = None
        self._backlog = 0
        self._window = 0
        self._last_change = 0
        self._last_log = 0.0
        self._last_floor: float | None = None
        self._last_prefill = 0.0
        self._last_dec_peak = 0
        # (bucket, lpt) -> (ivl, floor_est, prefill_rate)
        self._table: dict[tuple[tuple[str, int], int], tuple[float, float, float]] = {}
        self._blocked: dict[tuple[tuple[str, int], int], float] = {}
        self._recent_floors: deque[float] = deque(maxlen=3)

        self.log.info(
            "[dyn-prefill] ENABLED baseline lpt=%d ivl=%.1f -> start lpt=%d ivl=%.1f | "
            "ladder=%s solo=%d batch=%d shared=%d floor=%.1f t/s window=%.1fs "
            "ivl_max=%.1f headroom=%.2f dwell=%d fine=%d step=%.2f",
            base_lpt,
            base_ivl,
            self.lpt,
            self.ivl,
            self.ladder,
            self.solo_lpt,
            self.batch_lpt,
            self.safe_lpt,
            self.floor_tps,
            self.window_s,
            self.ivl_max,
            self.headroom,
            self.dwell,
            self.fine_ivl,
            self.ivl_step,
        )

    # ------------------------------------------------------------------ decisions
    def effective_lpt(self) -> int:
        """Chunk cap to use this step; replaces the configured value entirely."""
        return self.lpt

    def decide_defer(self, has_decoder: bool, capacity_bound: bool) -> bool:
        """Whether to defer new prefills this step.

        Never defers with no decoder to protect, and does not advance the cadence while
        the saturation guard disables throttling anyway, so the phase keeps meaning
        "steps since the last release".
        """
        if not has_decoder or capacity_bound:
            self._phase = 0.0
            return False
        self._phase += 1.0 / self.ivl
        if self._phase >= 1.0:
            self._phase -= 1.0
            return False  # release step
        return True

    # ---------------------------------------------------------------- observation
    def begin_step(self, dt: float) -> None:
        """Once per engine step, before scheduling."""
        dt = min(max(dt, 0.0), _MAX_STEP_S)
        self._dt = dt
        self._win_elapsed += dt
        if self._win_elapsed >= self.window_s:
            self._close_window()
        now = time.monotonic()
        if now - self._last_log >= self.log_every:
            self._last_log = now
            self._log_learn()

    def note_scheduled(
        self,
        decoders: set[str],
        prefill_tokens: int,
        backlog: int,
        inflight: int,
        queued: int,
    ) -> None:
        """Once per engine step, after scheduling."""
        self._win_steps += 1
        self._win_dec_peak = max(self._win_dec_peak, len(decoders))
        self._win_prefill += prefill_tokens
        self._backlog = backlog
        self._win_alone = self._win_alone and inflight <= 1 and queued == 0

        pending = inflight + queued
        if pending == 0:
            self._last_arrival = None  # drained: the next arrival starts a fresh shape
        elif pending > self._pending:
            now = time.monotonic()
            if (
                self._last_arrival is not None
                and now - self._last_arrival >= self.seq_gap_s
            ):
                self._win_sequential = True
            self._last_arrival = now
        self._pending = pending

        if decoders:
            for req_id in decoders:
                self._active[req_id] = self._active.get(req_id, 0.0) + self._dt

    def note_tokens(self, req_id: str, accepted: int) -> None:
        """Accepted decode tokens for one request (from update_from_output)."""
        if accepted:
            self._tokens[req_id] = self._tokens.get(req_id, 0) + accepted

    # ------------------------------------------------------------------- internals
    def _bucket(self) -> tuple[str, int]:
        n = self._win_dec_peak
        n_cls = "0" if n == 0 else ("1" if n == 1 else "2+")
        b = self._backlog
        return n_cls, 0 if b < 8000 else (1 if b < 64000 else 2)

    def _window_floor(self) -> float | None:
        """Worst per-decoder rate over the window, or None if no usable sample."""
        rates = [
            self._tokens.get(req_id, 0) / active
            for req_id, active in self._active.items()
            if active >= _MIN_SAMPLE_S
        ]
        return min(rates) if rates else None

    def _close_window(self) -> None:
        floor_est = self._window_floor()
        prefill_rate = (
            self._win_prefill / self._win_elapsed if self._win_elapsed else 0.0
        )
        self._last_floor = floor_est
        self._last_prefill = prefill_rate
        self._last_dec_peak = self._win_dec_peak
        if floor_est is not None:
            self._recent_floors.append(floor_est)
        self._window += 1
        bucket = self._bucket()

        # Shape: which measured cell this batch should be sitting in. The shape is an
        # episode property: once requests have arrived one after another it stays
        # conservative for as long as the box is shared, so a single quiet window cannot
        # talk the loop back into the cell that starves the decoder it just protected.
        entering_shared = not self._win_alone and not self._shared
        if self._win_alone:
            self._shared_sequential = False
            self._shared = False
        else:
            self._shared = True
            if self._win_sequential:
                self._shared_sequential = True
        ceiling = self.batch_lpt
        if self._shared_sequential:
            ceiling = min(ceiling, self.safe_lpt)
        reason = "shared_sequential" if self._shared_sequential else "shared_batch"
        # The interval is asserted at the ceiling when a sharing episode starts, or when
        # the ratchet is off. Inside an episode it belongs to the ratchet, so a window
        # with no decoder in it cannot undo a shortening that was measured safe.
        ivl_cell = self.ivl_max if (entering_shared or not self.fine_ivl) else self.ivl
        if self._win_dec_peak == 0:
            if self._win_alone:
                # Nothing to protect and nothing coming: uncapped with no cadence, i.e.
                # the static lpt=0 solo behaviour.
                self._apply(self.solo_lpt, 1.0, "solo", floor_est, prefill_rate, bucket)
            else:
                self._apply(ceiling, ivl_cell, reason, floor_est, prefill_rate, bucket)
        elif self._cap_rank(self.lpt) > self._cap_rank(ceiling):
            # A shared shape with a decoder already running and a cap bigger than the
            # shape allows (e.g. still at the solo cap): give it up now, not after the
            # first window it starves someone in.
            self._apply(ceiling, ivl_cell, reason, floor_est, prefill_rate, bucket)

        if floor_est is None:
            pass  # too few decode samples to judge; hold
        elif floor_est < self.floor_tps:
            self._blocked[(bucket, self.lpt)] = self.ivl
            if self.ivl < self.ivl_max:
                self._apply(
                    self.lpt,
                    self.ivl_max,
                    "floor_low_recover_ivl",
                    floor_est,
                    prefill_rate,
                    bucket,
                )
            else:
                lower = self._lower_rung()
                if lower is not None:
                    self._apply(
                        lower,
                        self.ivl,
                        "floor_low_recover_cap",
                        floor_est,
                        prefill_rate,
                        bucket,
                    )
        elif self._window - self._last_change < self.dwell:
            pass  # dwell: one good window is not evidence
        elif prefill_rate <= 0.0:
            # Nothing prefilled in that window: a bigger cap speeds nothing up here and
            # only arms the next turn's starvation.
            pass
        elif floor_est >= self.floor_tps * self.headroom:
            higher = self._higher_rung(ceiling)
            if higher is not None:
                self._apply(
                    higher, self.ivl, "slack_raise_cap", floor_est, prefill_rate, bucket
                )
            elif (
                self.fine_ivl
                and self.ivl > 1.0
                and min(self._recent_floors) >= self.floor_tps * self._ivl_margin
            ):
                # No rung left to raise, but the floor has room: give the cadence back
                # in small steps. The rolling minimum gates it -- one good window after
                # a bad one is not evidence, and the client's floor is a min over the
                # whole turn, so a window spent under it is permanent.
                self._apply(
                    self.lpt,
                    max(1.0, self.ivl - self.ivl_step),
                    "fine_shorten_ivl",
                    floor_est,
                    prefill_rate,
                    bucket,
                )

        if floor_est is not None and floor_est >= self.floor_tps:
            key = (bucket, self.lpt)
            best = self._table.get(key)
            if best is None or prefill_rate > best[2]:
                self._table[key] = (round(self.ivl, 1), floor_est, prefill_rate)

        self._active = {}
        self._tokens = {}
        self._win_prefill = 0
        self._win_elapsed = 0.0
        self._win_dec_peak = 0
        self._win_steps = 0
        self._win_alone = True
        self._win_sequential = False

    def _cap_rank(self, lpt: int) -> int:
        """Position in the bite-size ordering; 0 (uncapped) ranks last, i.e. biggest."""
        return self.ladder.index(lpt) if lpt in self.ladder else len(self.ladder)

    def _higher_rung(self, ceiling: int) -> int | None:
        """Next cap up, not above `ceiling` and not already known to break the floor."""
        ceiling_rank = self._cap_rank(ceiling)
        i = self._cap_rank(self.lpt)
        while i + 1 < len(self.ladder):
            nxt = self.ladder[i + 1]
            if ceiling and self._cap_rank(nxt) > ceiling_rank:
                return None
            failed_at = self._blocked.get((self._bucket(), nxt))
            if failed_at is None or self.ivl >= failed_at * 1.5:
                return nxt
            i += 1
        return None

    def _lower_rung(self) -> int | None:
        i = self._cap_rank(self.lpt)
        return self.ladder[i - 1] if i > 0 else None

    def _apply(
        self,
        lpt: int,
        ivl: float,
        reason: str,
        floor_est: float | None,
        prefill_rate: float,
        bucket: tuple[str, int],
    ) -> None:
        if lpt == self.lpt and abs(ivl - self.ivl) < 1e-9:
            return
        old_lpt, old_ivl = self.lpt, self.ivl
        self.lpt, self.ivl = lpt, ivl
        self._phase = 0.0  # restart the cadence so the new duty cycle applies at once
        self._last_change = self._window
        self.log.info(
            "[dyn-prefill] ADJUST %s: lpt %d->%d ivl %.1f->%.1f | floor=%s t/s (floor "
            "target %.1f) prefill=%.0f t/s decoders=%d backlog=%d bucket=%s",
            reason,
            old_lpt,
            lpt,
            old_ivl,
            ivl,
            "n/a" if floor_est is None else f"{floor_est:.1f}",
            self.floor_tps,
            prefill_rate,
            self._win_dec_peak,
            self._backlog,
            bucket,
        )

    def _log_learn(self) -> None:
        # One ADJUST-independent line per interval, so the running state and what has
        # been learned are both in the log even when nothing changes.
        self.log.info(
            "[dyn-prefill] LEARN live lpt=%d ivl=%.1f floor=%s t/s floor_min=%s "
            "prefill=%.0f t/s decoders=%d backlog=%d windows=%d",
            self.lpt,
            self.ivl,
            "n/a" if self._last_floor is None else f"{self._last_floor:.1f}",
            "n/a" if not self._recent_floors else f"{min(self._recent_floors):.1f}",
            self._last_prefill,
            self._last_dec_peak,
            self._backlog,
            self._window,
        )
        for ((n_cls, b_cls), lpt), (ivl, floor_est, prefill_rate) in sorted(
            self._table.items()
        ):
            self.log.info(
                "[dyn-prefill] LEARN best decoders>=%s backlog_class=%d lpt=%d: "
                "ivl=%.1f floor=%.1f t/s prefill=%.0f t/s",
                n_cls,
                b_cls,
                lpt,
                ivl,
                floor_est,
                prefill_rate,
            )
        for ((n_cls, b_cls), lpt), failed_at in sorted(self._blocked.items()):
            self.log.info(
                "[dyn-prefill] LEARN blocked decoders>=%s backlog_class=%d: "
                "lpt=%d broke the floor at ivl<=%.1f",
                n_cls,
                b_cls,
                lpt,
                failed_at,
            )


def maybe_create(base_lpt: int, base_ivl: int, logger: logging.Logger):
    """Build the controller when opted in, else None (the static path)."""
    if os.environ.get("VLLM_RDNA_DYNAMIC_PREFILL", "0") != "1":
        return None
    return DynamicPrefillController(base_lpt, base_ivl, logger)
