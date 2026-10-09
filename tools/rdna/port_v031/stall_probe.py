#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Mixed-batch stall probe: decoders vs. long prefills, per-token timelines.

    python stall_probe.py --base-url http://127.0.0.1:18120 --model flash-next \
        --decoders 8 --prefill-lens 4096,16384 --repeats 3 --out DIR \
        [--server-log serve.log] [--max-stall-ms 1500 ...]

Runs against a live OpenAI-compatible server (streaming ``/v1/completions``)
and measures how long-prompt prefills and running decoders get in each
other's way. Scenarios (``--scenarios``, comma list, run in this order):

  solo      each prefill length alone (no decoders) -> reference TTFT
  inject    N decoders reach steady decode, a baseline window is recorded,
            then each prefill length is injected one at a time (wait for
            its first token + a recovery window before the next one)
  periodic  N steady decoders; one prefill of ``--periodic-len`` every
            ``--period-s`` seconds, ``--period-count`` times (may overlap)
  reverse   one long prefill first; ``--reverse-delay-s`` later the N
            decoders arrive while it is still prefilling

For every injection window (prefill submit -> its first token) it reports,
over the decoder streams: inter-token gaps p50/p90/p99/max vs the baseline
window, the longest stall, the total stalled time (gaps above
``--stall-factor`` x baseline ITL), decode tokens produced during the
prefill, per-decoder fairness (min/max tokens, Jain index, starved streams),
and the injected request's TTFT vs its solo TTFT. Each scenario runs
``--repeats`` times; the summary gives mean/sd/min/max across repeats.

Determinism: prompts are token-id lists of exact length, built from a fixed
corpus tokenized once through the server's ``/tokenize`` (synthetic ids if
that endpoint is missing). Every request gets a distinct 16-token header
derived from (``--seed``, ``--salt``, scenario, repeat, stream), so prefix
caching never hides prefill work. ``--salt`` defaults to a per-invocation
value; pin it (on a fresh server or with prefix caching off) for bitwise
identical prompts across invocations. Injections are scheduled at fixed
offsets from the moment every decoder has ``--warmup-tokens`` tokens.

Token counts come from ``stream_options.continuous_usage_stats`` (exact with
MTP / multi-token chunks); timestamps are ``time.monotonic()``.

``--server-log`` (optional): a serve log with ``--enable-logging-iteration-
details`` lines. Each window is then annotated with the scheduler steps that
ran in it (prefill vs decode tokens per step, step time, steps that left
decoders out). Log stamps have 1 s resolution, so this is +-1 s.

Outputs in ``--out``: ``stall_probe.json`` (config, per-run windows,
aggregates, verdict), ``summary.txt`` and, with ``--timeline``,
``timeline.csv`` (one row per streamed chunk and per event, for plotting).

Exit code: 0 = pass (or no gates set), 1 = a gate failed, 2 = request
errors / no steady state. Stdlib only.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import random
import re
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

# Fixed, self-written corpus: tokenized once, repeated to exact lengths.
CORPUS = (
    "The build server sits in a cold room at the end of the corridor. Eight "
    "graphics cards share two PCIe switches, and every request that reaches the "
    "inference server is split into chunks, scheduled, and executed in steps. A "
    "long prompt is read in pieces of a few thousand tokens; short requests that "
    "are already generating text wait for their next token while the piece is "
    "processed. Engineers measure the time between tokens, the time to the first "
    "token, and how many tokens each stream receives while a large prompt is "
    "being read. They write the numbers into tables, compare them with the "
    "previous week, and argue about thresholds. The scheduler decides, step by "
    "step, how many prompt tokens and how many generated tokens go into each "
    "batch. When the batch holds a large prompt chunk, the step takes longer, "
    "and every stream in that batch waits for it to finish. When the scheduler "
    "defers the prompt, the generating streams move quickly but the prompt "
    "waits instead. Neither choice is free, and the right balance depends on "
    "the workload, the model, the memory budget, and what the users notice "
    "first. A careful measurement repeats the same experiment several times, "
    "with the same prompts, at the same moments, and reports how much the "
    "results move from run to run before anyone trusts a single number. "
)

ITER_RE = re.compile(
    r"(\d\d-\d\d \d\d:\d\d:\d\d).*?Iteration\((\d+)\): (\d+) context requests, "
    r"(\d+) context tokens, (\d+) generation requests, (\d+) generation tokens, "
    r"iteration elapsed time: ([\d.]+) ms"
)


# --------------------------------------------------------------------------
# Statistics helpers (pure)
# --------------------------------------------------------------------------


def percentile(values: list[float], q: float) -> float:
    """Linear-interpolated percentile, q in [0, 100]. NaN for empty input."""
    if not values:
        return math.nan
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    pos = (len(s) - 1) * q / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def dist(values_s: list[float]) -> dict[str, float]:
    """p50/p90/p99/max/mean in ms of a list of durations given in seconds."""
    ms = [v * 1000.0 for v in values_s]
    return {
        "n": len(ms),
        "p50": percentile(ms, 50),
        "p90": percentile(ms, 90),
        "p99": percentile(ms, 99),
        "max": max(ms) if ms else math.nan,
        "mean": statistics.fmean(ms) if ms else math.nan,
    }


def jain(values: list[float]) -> float:
    """Jain fairness index: 1 = perfectly even, 1/n = one stream gets all."""
    if not values:
        return math.nan
    sq = sum(v * v for v in values)
    if sq == 0:
        return math.nan
    return sum(values) ** 2 / (len(values) * sq)


def spread(values: list[float]) -> dict[str, float]:
    vals = [v for v in values if v is not None and not math.isnan(v)]
    if not vals:
        return {
            "n": 0,
            "mean": math.nan,
            "sd": math.nan,
            "min": math.nan,
            "max": math.nan,
            "median": math.nan,
        }
    return {
        "n": len(vals),
        "mean": statistics.fmean(vals),
        "sd": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        "min": min(vals),
        "max": max(vals),
        "median": statistics.median(vals),
    }


# --------------------------------------------------------------------------
# Stream records
# --------------------------------------------------------------------------


@dataclass
class Stream:
    """One streamed request. Times are seconds relative to the probe clock."""

    sid: str
    role: str  # "decoder" | "inject"
    prompt_tokens: int
    max_tokens: int
    submit: float = math.nan
    chunks: list[tuple[float, int]] = field(default_factory=list)
    end: float = math.nan
    error: str | None = None
    usage_prompt: int | None = None
    usage_completion: int | None = None
    finish_reason: str | None = None
    stopped_by_probe: bool = False

    @property
    def first(self) -> float:
        return self.chunks[0][0] if self.chunks else math.nan

    @property
    def ttft(self) -> float:
        return self.first - self.submit

    @property
    def tokens(self) -> int:
        return sum(n for _, n in self.chunks)

    def gaps(self) -> list[tuple[float, float]]:
        """(t_prev, t_cur) intervals between consecutive token chunks."""
        return [
            (self.chunks[i - 1][0], self.chunks[i][0])
            for i in range(1, len(self.chunks))
        ]


def tokens_between(s: Stream, t0: float, t1: float) -> int:
    return sum(n for t, n in s.chunks if t0 < t <= t1)


def overlapping_gaps(
    s: Stream, t0: float, t1: float, include_ttft: bool = False
) -> list[tuple[float, float]]:
    """Gaps whose interval intersects [t0, t1]. A stream that stayed open past
    t1 without another token contributes the open interval (last, end].
    With include_ttft, the wait for the first token counts as a gap too."""
    gaps = s.gaps()
    if include_ttft:
        first = s.first if s.chunks else s.end
        gaps = [(s.submit, first), *gaps]
    out = [(a, b) for a, b in gaps if b > t0 and a < t1]
    if s.chunks and s.chunks[-1][0] < t1 and s.end > t1:
        out.append((s.chunks[-1][0], s.end))
    return out


# --------------------------------------------------------------------------
# Window analysis (pure; unit-tested)
# --------------------------------------------------------------------------


def baseline_stats(decoders: list[Stream], t0: float, t1: float) -> dict[str, Any]:
    """Decoder behaviour in an undisturbed window [t0, t1]."""
    gaps = [b - a for s in decoders for a, b in s.gaps() if a >= t0 and b <= t1]
    toks = [tokens_between(s, t0, t1) for s in decoders]
    dur = max(t1 - t0, 1e-9)
    return {
        "t0": t0,
        "t1": t1,
        "window_s": dur,
        "gaps_ms": dist(gaps),
        "decode_tokens_total": sum(toks),
        "decode_tps_total": sum(toks) / dur,
        "decode_tps_per_decoder": [t / dur for t in toks],
    }


def window_stats(
    decoders: list[Stream],
    t0: float,
    t1: float,
    baseline: dict[str, Any],
    stall_factor: float,
    stall_min_ms: float,
    include_ttft: bool = False,
) -> dict[str, Any]:
    """Decoder behaviour while a prefill is in flight over [t0, t1]."""
    dur = max(t1 - t0, 1e-9)
    base_p50 = baseline["gaps_ms"]["p50"]
    thr_ms = max(
        stall_factor * base_p50 if not math.isnan(base_p50) else 0.0, stall_min_ms
    )
    thr = thr_ms / 1000.0
    all_gaps: list[float] = []
    per = []
    for s in decoders:
        gaps = overlapping_gaps(s, t0, t1, include_ttft)
        lens = [b - a for a, b in gaps]
        all_gaps.extend(lens)
        stalled = [(a, b) for a, b in gaps if b - a > thr]
        stalled_s = sum(b - a for a, b in stalled)
        frozen_s = sum(max(0.0, min(b, t1) - max(a, t0)) for a, b in stalled)
        per.append(
            {
                "sid": s.sid,
                "tokens": tokens_between(s, t0, t1),
                "max_gap_ms": max(lens) * 1000.0 if lens else math.nan,
                "stalled_s": stalled_s,
                "frozen_frac": frozen_s / dur,
                "n_stalls": len(stalled),
            }
        )
    toks = [p["tokens"] for p in per]
    base_tps = baseline["decode_tps_total"]
    win_tps = sum(toks) / dur
    starve_s = max(thr, 0.0)
    return {
        "t0": t0,
        "t1": t1,
        "window_s": dur,
        "stall_threshold_ms": thr_ms,
        "gaps_ms": dist(all_gaps),
        "max_gap_ms": max(all_gaps) * 1000.0 if all_gaps else math.nan,
        "stalled_s_max": max((p["stalled_s"] for p in per), default=math.nan),
        "stalled_s_mean": statistics.fmean(p["stalled_s"] for p in per)
        if per
        else math.nan,
        "frozen_frac_max": max((p["frozen_frac"] for p in per), default=math.nan),
        "decode_tokens_total": sum(toks),
        "decode_tokens_min": min(toks) if toks else 0,
        "decode_tokens_max": max(toks) if toks else 0,
        "decode_tps_window": win_tps,
        "decode_tps_baseline": base_tps,
        "decode_retention": win_tps / base_tps if base_tps > 0 else math.nan,
        "jain": jain([float(t) for t in toks]),
        "starved": sum(1 for t in toks if t == 0) if dur > starve_s else 0,
        "per_decoder": per,
    }


# --------------------------------------------------------------------------
# Server log corroboration (pure)
# --------------------------------------------------------------------------


@dataclass
class Iteration:
    wall: int  # epoch second of the log line
    index: int
    ctx_reqs: int
    ctx_tokens: int
    gen_reqs: int
    gen_tokens: int
    elapsed_ms: float


def parse_iteration_log(path: str, year: int) -> list[Iteration]:
    out: list[Iteration] = []
    try:
        with open(path, errors="replace") as f:
            for line in f:
                if "Iteration(" not in line:
                    continue
                m = ITER_RE.search(line)
                if not m:
                    continue
                stamp = dt.datetime.strptime(
                    f"{year}-{m.group(1)}", "%Y-%m-%d %H:%M:%S"
                )
                out.append(
                    Iteration(
                        wall=int(stamp.timestamp()),
                        index=int(m.group(2)),
                        ctx_reqs=int(m.group(3)),
                        ctx_tokens=int(m.group(4)),
                        gen_reqs=int(m.group(5)),
                        gen_tokens=int(m.group(6)),
                        elapsed_ms=float(m.group(7)),
                    )
                )
    except OSError:
        return []
    return out


def server_window(
    iters: list[Iteration], wall0: float, wall1: float, n_decoders: int
) -> dict[str, Any] | None:
    lo, hi = math.floor(wall0), math.floor(wall1)
    sel = [it for it in iters if lo <= it.wall <= hi]
    if not sel:
        return None
    mixed = [it for it in sel if it.ctx_tokens > 0 and it.gen_reqs > 0]
    pre = [it for it in sel if it.ctx_tokens > 0 and it.gen_reqs == 0]
    dec = [it for it in sel if it.ctx_tokens == 0 and it.gen_reqs > 0]

    def mean_ms(xs: list[Iteration]) -> float:
        return statistics.fmean(x.elapsed_ms for x in xs) if xs else math.nan

    return {
        "steps": len(sel),
        "steps_mixed": len(mixed),
        "steps_prefill_only": len(pre),
        "steps_decode_only": len(dec),
        "prefill_tokens": sum(it.ctx_tokens for it in sel),
        "decode_tokens": sum(it.gen_tokens for it in sel),
        "max_step_ms": max(it.elapsed_ms for it in sel),
        "mean_step_ms_mixed": mean_ms(mixed),
        "mean_step_ms_prefill_only": mean_ms(pre),
        "mean_step_ms_decode_only": mean_ms(dec),
        "max_prefill_tokens_per_step": max(it.ctx_tokens for it in sel),
        "steps_missing_decoders": sum(1 for it in sel if it.gen_reqs < n_decoders),
        "step_time_sum_s": sum(it.elapsed_ms for it in sel) / 1000.0,
    }


# --------------------------------------------------------------------------
# HTTP client
# --------------------------------------------------------------------------


class Client:
    def __init__(self, base_url: str, model: str, timeout: float) -> None:
        self.base = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def _post(self, path: str, body: dict, timeout: float | None = None):
        req = urllib.request.Request(
            self.base + path,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        return urllib.request.urlopen(req, timeout=timeout or self.timeout)

    def tokenize(self, text: str) -> list[int] | None:
        try:
            with self._post(
                "/tokenize",
                {"model": self.model, "prompt": text, "add_special_tokens": False},
                timeout=60,
            ) as r:
                toks = json.loads(r.read()).get("tokens")
            return [int(t) for t in toks] if toks else None
        except (urllib.error.URLError, OSError, ValueError):
            return None

    def running_requests(self) -> float | None:
        """vllm:num_requests_running + waiting from /metrics (None if absent)."""
        try:
            with urllib.request.urlopen(self.base + "/metrics", timeout=10) as r:
                text = r.read().decode(errors="replace")
        except (urllib.error.URLError, OSError):
            return None
        total, seen = 0.0, False
        for line in text.splitlines():
            if line.startswith(
                ("vllm:num_requests_running", "vllm:num_requests_waiting")
            ):
                try:
                    total += float(line.rsplit(" ", 1)[1])
                    seen = True
                except ValueError:
                    pass
        return total if seen else None

    def stream(
        self,
        s: Stream,
        prompt: list[int],
        clock: Clock,
        stop: threading.Event | None,
    ) -> None:
        body = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": s.max_tokens,
            "temperature": 0,
            "ignore_eos": True,
            "stream": True,
            "stream_options": {"include_usage": True, "continuous_usage_stats": True},
        }
        if math.isnan(s.submit):
            s.submit = clock.now()
        prev = 0
        try:
            with self._post("/v1/completions", body) as resp:
                for raw in resp:
                    line = raw.decode(errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    t = clock.now()
                    chunk = json.loads(payload)
                    usage = chunk.get("usage") or {}
                    choices = chunk.get("choices") or []
                    if usage.get("prompt_tokens") is not None:
                        s.usage_prompt = usage["prompt_tokens"]
                    if not choices:
                        if usage.get("completion_tokens") is not None:
                            s.usage_completion = usage["completion_tokens"]
                        continue
                    ch = choices[0]
                    if ch.get("finish_reason"):
                        s.finish_reason = ch["finish_reason"]
                    if usage.get("completion_tokens") is not None:
                        n = int(usage["completion_tokens"]) - prev
                        prev = int(usage["completion_tokens"])
                    else:
                        n = 1 if ch.get("text") else 0
                    if n > 0:
                        s.chunks.append((t, n))
                    if stop is not None and stop.is_set():
                        s.stopped_by_probe = True
                        break
        except (urllib.error.URLError, OSError, ValueError) as e:
            s.error = f"{type(e).__name__}: {e}"
        s.end = clock.now()


class Clock:
    """Monotonic probe clock with a wall-clock anchor for log correlation."""

    def __init__(self) -> None:
        self.mono0 = time.monotonic()
        self.wall0 = time.time()

    def now(self) -> float:
        return time.monotonic() - self.mono0

    def wall(self, t: float) -> float:
        return self.wall0 + t

    def sleep_until(self, t: float) -> None:
        d = t - self.now()
        if d > 0:
            time.sleep(d)


# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------


class Prompts:
    """Token-exact prompts with a per-request unique header."""

    HEADER = 16

    def __init__(self, pool: list[int], seed: int, salt: str) -> None:
        if len(pool) < 32:
            raise ValueError("token pool too small")
        self.pool = pool
        self.seed = seed
        self.salt = salt

    def make(self, n_tokens: int, key: str) -> list[int]:
        digest = hashlib.sha256(f"{self.seed}|{self.salt}|{key}".encode()).digest()
        rng = random.Random(int.from_bytes(digest[:8], "little"))
        n_head = min(self.HEADER, n_tokens)
        head = [rng.choice(self.pool) for _ in range(n_head)]
        start = rng.randrange(len(self.pool))
        body = [
            self.pool[(start + i) % len(self.pool)] for i in range(n_tokens - n_head)
        ]
        return head + body


def build_pool(client: Client, corpus: str, seed: int) -> tuple[list[int], str]:
    toks = client.tokenize(corpus)
    if toks and len(toks) >= 32:
        return toks, "tokenize"
    rng = random.Random(seed)
    return [rng.randrange(1000, 20000) for _ in range(512)], "synthetic"


# --------------------------------------------------------------------------
# Scenario runner
# --------------------------------------------------------------------------


@dataclass
class Run:
    scenario: str
    repeat: int
    streams: list[Stream] = field(default_factory=list)
    events: list[tuple[str, float, str]] = field(default_factory=list)
    baseline: tuple[float, float] | None = None
    error: str | None = None

    @property
    def decoders(self) -> list[Stream]:
        return [s for s in self.streams if s.role == "decoder"]

    @property
    def injects(self) -> list[Stream]:
        return [s for s in self.streams if s.role == "inject"]


class Probe:
    def __init__(self, args: argparse.Namespace, client: Client, prompts: Prompts):
        self.a = args
        self.client = client
        self.prompts = prompts
        self.clock = Clock()

    # -- request helpers --------------------------------------------------
    def _launch(
        self,
        run: Run,
        role: str,
        idx: int,
        n_prompt: int,
        max_tokens: int,
        stop: threading.Event | None,
    ) -> tuple[Stream, threading.Thread]:
        sid = f"{role[:3]}{idx}"
        key = f"{run.scenario}|{run.repeat}|{sid}"
        s = Stream(sid=sid, role=role, prompt_tokens=n_prompt, max_tokens=max_tokens)
        prompt = self.prompts.make(n_prompt, key)
        s.submit = self.clock.now()
        th = threading.Thread(
            target=self.client.stream, args=(s, prompt, self.clock, stop), daemon=True
        )
        run.streams.append(s)
        th.start()
        return s, th

    def _start_decoders(self, run: Run, stop: threading.Event):
        out = []
        for i in range(self.a.decoders):
            out.append(
                self._launch(
                    run,
                    "decoder",
                    i,
                    self.a.decoder_prompt_tokens,
                    self.a.decoder_max_tokens,
                    stop,
                )
            )
            if self.a.decoder_stagger_s > 0:
                time.sleep(self.a.decoder_stagger_s)
        return out

    def _wait_steady(self, run: Run, decs: list[Stream]) -> float | None:
        deadline = self.clock.now() + self.a.steady_timeout_s
        while self.clock.now() < deadline:
            if any(s.error for s in decs):
                run.error = "decoder error: " + next(s.error for s in decs if s.error)
                return None
            if all(s.tokens >= self.a.warmup_tokens for s in decs):
                t = self.clock.now()
                run.events.append(("steady", t, ""))
                return t
            time.sleep(0.01)
        run.error = "decoders never reached steady decode"
        return None

    def _inject(
        self, run: Run, idx: int, n_prompt: int
    ) -> tuple[Stream, threading.Thread]:
        s, th = self._launch(
            run, "inject", idx, n_prompt, self.a.inject_max_tokens, None
        )
        run.events.append(("inject_submit", s.submit, f"{s.sid}:{n_prompt}"))
        return s, th

    def _stop(self, stop: threading.Event, threads: list[threading.Thread]) -> None:
        stop.set()
        for th in threads:
            th.join(timeout=self.a.request_timeout_s)

    def wait_idle(self) -> None:
        """Let aborted streams drain before the next run (metrics, else sleep)."""
        deadline = time.monotonic() + 120
        time.sleep(self.a.settle_s)
        while time.monotonic() < deadline:
            n = self.client.running_requests()
            if n is None or n == 0:
                return
            time.sleep(0.5)

    # -- scenarios --------------------------------------------------------
    def solo(self, repeat: int) -> Run:
        run = Run("solo", repeat)
        for i, n in enumerate(self.a.prefill_lens):
            s, th = self._inject(run, i, n)
            th.join(timeout=self.a.request_timeout_s)
            run.events.append(("inject_first", s.first, s.sid))
            self.wait_idle()
        return run

    def inject(self, repeat: int) -> Run:
        run = Run("inject", repeat)
        stop = threading.Event()
        decs = self._start_decoders(run, stop)
        threads = [th for _, th in decs]
        t_steady = self._wait_steady(run, [s for s, _ in decs])
        if t_steady is None:
            self._stop(stop, threads)
            return run
        t_next = t_steady + self.a.baseline_s
        run.baseline = (t_steady, t_next)
        for i, n in enumerate(self.a.prefill_lens):
            self.clock.sleep_until(t_next)
            s, th = self._inject(run, i, n)
            th.join(timeout=self.a.request_timeout_s)
            threads.append(th)
            run.events.append(("inject_first", s.first, s.sid))
            ref = s.first if not math.isnan(s.first) else self.clock.now()
            t_next = ref + self.a.recover_s
        self.clock.sleep_until(t_next)
        self._stop(stop, threads)
        return run

    def periodic(self, repeat: int) -> Run:
        run = Run("periodic", repeat)
        stop = threading.Event()
        decs = self._start_decoders(run, stop)
        threads = [th for _, th in decs]
        t_steady = self._wait_steady(run, [s for s, _ in decs])
        if t_steady is None:
            self._stop(stop, threads)
            return run
        t_base = t_steady + self.a.baseline_s
        run.baseline = (t_steady, t_base)
        injs = []
        for k in range(self.a.period_count):
            self.clock.sleep_until(t_base + k * self.a.period_s)
            injs.append(self._inject(run, k, self.a.periodic_len))
        for s, th in injs:
            th.join(timeout=self.a.request_timeout_s)
            run.events.append(("inject_first", s.first, s.sid))
        threads += [th for _, th in injs]
        self.clock.sleep_until(self.clock.now() + self.a.recover_s)
        self._stop(stop, threads)
        return run

    def reverse(self, repeat: int) -> Run:
        run = Run("reverse", repeat)
        stop = threading.Event()
        s_inj, th_inj = self._inject(run, 0, self.a.reverse_len)
        self.clock.sleep_until(s_inj.submit + self.a.reverse_delay_s)
        t_dec = self.clock.now()
        run.events.append(("decoders_submit", t_dec, ""))
        decs = self._start_decoders(run, stop)
        th_inj.join(timeout=self.a.request_timeout_s)
        run.events.append(("inject_first", s_inj.first, s_inj.sid))
        threads = [th for _, th in decs] + [th_inj]
        self._wait_steady(run, [s for s, _ in decs])
        self.clock.sleep_until(self.clock.now() + self.a.recover_s)
        self._stop(stop, threads)
        return run


# --------------------------------------------------------------------------
# Analysis of runs
# --------------------------------------------------------------------------


def analyze_run(
    run: Run,
    args: argparse.Namespace,
    solo_ttft: dict[int, float],
    ref_dec_ttft: float,
    iters: list[Iteration],
    clock_wall0: float,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "scenario": run.scenario,
        "repeat": run.repeat,
        "error": run.error,
    }
    errs = [f"{s.sid}: {s.error}" for s in run.streams if s.error]
    if errs:
        out["stream_errors"] = errs
    exhausted = [s.sid for s in run.decoders if not s.stopped_by_probe and not s.error]
    if exhausted:
        out["decoders_finished_early"] = exhausted
    if run.scenario == "solo":
        out["solo"] = [
            {
                "len": s.prompt_tokens,
                "ttft_s": s.ttft,
                "prefill_tps": s.prompt_tokens / s.ttft if s.ttft > 0 else math.nan,
                "usage_prompt": s.usage_prompt,
            }
            for s in run.injects
        ]
        return out
    decs = run.decoders
    if run.scenario in ("inject", "periodic") and run.baseline:
        base = baseline_stats(decs, *run.baseline)
    else:
        base = None
    out["baseline"] = base
    windows = []
    if run.scenario == "reverse":
        inj = run.injects[0] if run.injects else None
        t_dec = next((t for n, t, _ in run.events if n == "decoders_submit"), math.nan)
        if inj is not None and not math.isnan(inj.first):
            # Baseline for gaps: decoders after the prefill finished.
            tail0 = inj.first
            tail1 = max((s.end for s in decs), default=tail0)
            base = baseline_stats(decs, tail0, tail1)
            out["baseline"] = base
            w = window_stats(
                decs,
                t_dec,
                inj.first,
                base,
                args.stall_factor,
                args.stall_min_ms,
                include_ttft=True,
            )
            dttft = [s.ttft for s in decs]
            w.update(
                {
                    "len": inj.prompt_tokens,
                    "ttft_s": inj.ttft,
                    "solo_ttft_s": solo_ttft.get(inj.prompt_tokens, math.nan),
                    "decoder_ttft_s": spread(dttft),
                    "decoder_ttft_ref_s": ref_dec_ttft,
                    "decoders_served_during_prefill": sum(
                        1 for s in decs if s.first < inj.first
                    ),
                }
            )
            w["slowdown"] = (
                w["ttft_s"] / w["solo_ttft_s"] if w["solo_ttft_s"] else math.nan
            )
            w["prefill_tps"] = (
                inj.prompt_tokens / inj.ttft if inj.ttft > 0 else math.nan
            )
            if iters:
                w["server"] = server_window(
                    iters, clock_wall0 + t_dec, clock_wall0 + inj.first, len(decs)
                )
            windows.append(w)
        out["windows"] = windows
        return out
    if base is None:
        out["windows"] = []
        return out
    for inj in run.injects:
        if math.isnan(inj.first):
            continue
        w = window_stats(
            decs, inj.submit, inj.first, base, args.stall_factor, args.stall_min_ms
        )
        solo = solo_ttft.get(inj.prompt_tokens, math.nan)
        w.update(
            {
                "sid": inj.sid,
                "len": inj.prompt_tokens,
                "ttft_s": inj.ttft,
                "solo_ttft_s": solo,
                "slowdown": inj.ttft / solo
                if solo and not math.isnan(solo)
                else math.nan,
                "prefill_tps": inj.prompt_tokens / inj.ttft
                if inj.ttft > 0
                else math.nan,
            }
        )
        if iters:
            w["server"] = server_window(
                iters, clock_wall0 + inj.submit, clock_wall0 + inj.first, len(decs)
            )
        windows.append(w)
    if run.scenario == "periodic" and run.injects:
        firsts = [i.first for i in run.injects if not math.isnan(i.first)]
        if firsts:
            t0 = min(i.submit for i in run.injects)
            t1 = max(firsts)
            busy = window_stats(
                decs, t0, t1, base, args.stall_factor, args.stall_min_ms
            )
            busy.pop("per_decoder", None)
            out["busy"] = busy
    out["windows"] = windows
    return out


KEY_METRICS = [
    ("max_gap_ms", "max gap ms"),
    ("gap_p99_ms", "gap p99 ms"),
    ("gap_p50_ms", "gap p50 ms"),
    ("stalled_s_max", "stalled s (worst dec)"),
    ("frozen_frac_max", "frozen frac (worst)"),
    ("decode_tokens_total", "dec tokens in window"),
    ("decode_tokens_min", "dec tokens (min/dec)"),
    ("decode_retention", "dec rate vs baseline"),
    ("jain", "jain"),
    ("starved", "starved decoders"),
    ("ttft_s", "inject TTFT s"),
    ("slowdown", "TTFT / solo"),
    ("window_s", "window s"),
]


def flat(w: dict[str, Any]) -> dict[str, float]:
    f = {k: w.get(k, math.nan) for k, _ in KEY_METRICS}
    f["gap_p99_ms"] = w["gaps_ms"]["p99"]
    f["gap_p50_ms"] = w["gaps_ms"]["p50"]
    return f


def aggregate(analyses: list[dict[str, Any]]) -> dict[str, Any]:
    """Group windows by (scenario, len, slot) across repeats -> spreads."""
    groups: dict[str, list[dict[str, float]]] = {}
    base_groups: dict[str, list[float]] = {}
    solo: dict[int, list[float]] = {}
    for a in analyses:
        if a["scenario"] == "solo":
            for r in a.get("solo", []):
                solo.setdefault(r["len"], []).append(r["ttft_s"])
            continue
        if a.get("baseline") and a["scenario"] != "reverse":
            base_groups.setdefault(a["scenario"], []).append(
                a["baseline"]["gaps_ms"]["p50"]
            )
        for i, w in enumerate(a.get("windows", [])):
            slot = f"#{i}" if a["scenario"] == "periodic" else ""
            key = f"{a['scenario']}:{w['len']}{slot}"
            groups.setdefault(key, []).append(flat(w))
        if a.get("busy"):
            groups.setdefault(f"{a['scenario']}:busy", []).append(
                flat({**a["busy"], "len": 0})
            )
    agg: dict[str, Any] = {
        "solo_ttft_s": {str(k): spread(v) for k, v in solo.items()},
        "baseline_gap_p50_ms": {k: spread(v) for k, v in base_groups.items()},
        "windows": {},
    }
    for key, rows in groups.items():
        agg["windows"][key] = {m: spread([r[m] for r in rows]) for m, _ in KEY_METRICS}
    return agg


# --------------------------------------------------------------------------
# Gates
# --------------------------------------------------------------------------


def evaluate_gates(
    agg: dict[str, Any], args: argparse.Namespace
) -> list[dict[str, Any]]:
    stat = args.gate_stat
    checks = []

    def pick(sp: dict[str, float], worst_is_max: bool) -> float:
        if stat == "worst":
            return sp["max"] if worst_is_max else sp["min"]
        return sp["median"]

    for key, m in agg["windows"].items():
        scen = key.split(":")[0]
        gated = scen in ("inject", "periodic") or (
            scen == "reverse" and args.gate_reverse
        )
        if not gated:
            continue
        if args.max_stall_ms is not None:
            v = pick(m["max_gap_ms"], True)
            checks.append(
                {
                    "window": key,
                    "gate": "max_stall_ms",
                    "value": v,
                    "limit": args.max_stall_ms,
                    "ok": v <= args.max_stall_ms,
                }
            )
        if args.max_stalled_s is not None:
            v = pick(m["stalled_s_max"], True)
            checks.append(
                {
                    "window": key,
                    "gate": "max_stalled_s",
                    "value": v,
                    "limit": args.max_stalled_s,
                    "ok": v <= args.max_stalled_s,
                }
            )
        if args.min_decode_tokens_during_prefill is not None and not key.endswith(
            "busy"
        ):
            v = pick(m["decode_tokens_min"], False)
            lim = args.min_decode_tokens_during_prefill
            checks.append(
                {
                    "window": key,
                    "gate": "min_decode_tokens_during_prefill",
                    "value": v,
                    "limit": lim,
                    "ok": v >= lim,
                }
            )
        if args.max_ttft_slowdown is not None and not key.endswith("busy"):
            v = pick(m["slowdown"], True)
            if not math.isnan(v):
                checks.append(
                    {
                        "window": key,
                        "gate": "max_ttft_slowdown",
                        "value": v,
                        "limit": args.max_ttft_slowdown,
                        "ok": v <= args.max_ttft_slowdown,
                    }
                )
        if args.max_starved is not None:
            v = pick(m["starved"], True)
            checks.append(
                {
                    "window": key,
                    "gate": "max_starved",
                    "value": v,
                    "limit": args.max_starved,
                    "ok": v <= args.max_starved,
                }
            )
    return checks


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def fmt(v: float, nd: int = 1) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-"
    return f"{v:.{nd}f}"


def summary_text(result: dict[str, Any]) -> str:
    agg = result["aggregate"]
    c = result["config"]
    head = (
        f"stall_probe: {c['base_url']} model={c['model']} decoders={c['decoders']} "
        f"repeats={c['repeats']} prompts={result['prompt_source']} salt={c['salt']}"
    )
    lines = [head]
    for k, sp in agg["solo_ttft_s"].items():
        lines.append(
            f"  solo {k:>6} tok: TTFT {fmt(sp['median'], 2)} s "
            f"(sd {fmt(sp['sd'], 2)}, n={sp['n']})"
        )
    for k, sp in agg["baseline_gap_p50_ms"].items():
        lines.append(
            f"  baseline ITL p50 ({k}): {fmt(sp['median'])} ms "
            f"(sd {fmt(sp['sd'])}, n={sp['n']})"
        )
    hdr = (
        f"  {'window':<22}{'maxgap ms':>18}{'p99 ms':>9}{'p50 ms':>8}"
        f"{'stall s':>14}{'frz':>6}{'dec tok':>14}{'min/dec':>8}"
        f"{'retain':>7}{'jain':>6}{'starv':>6}{'TTFT s':>14}{'x solo':>7}"
    )
    lines.append(hdr)
    for key, m in agg["windows"].items():

        def ms(name: str, nd: int = 0, m: dict = m) -> str:
            return f"{fmt(m[name]['median'], nd)}±{fmt(m[name]['sd'], nd)}"

        def med(name: str, nd: int = 0, m: dict = m) -> str:
            return fmt(m[name]["median"], nd)

        lines.append(
            f"  {key:<22}{ms('max_gap_ms'):>18}{med('gap_p99_ms'):>9}"
            f"{med('gap_p50_ms'):>8}{ms('stalled_s_max', 2):>14}"
            f"{med('frozen_frac_max', 2):>6}{ms('decode_tokens_total'):>14}"
            f"{med('decode_tokens_min'):>8}{med('decode_retention', 2):>7}"
            f"{med('jain', 2):>6}{fmt(m['starved']['max'], 0):>6}"
            f"{ms('ttft_s', 2):>14}{med('slowdown', 2):>7}"
        )
    srv = []
    for a in result["runs"]:
        for w in a.get("windows", []):
            s = w.get("server")
            if s:
                srv.append((a["scenario"], a["repeat"], w["len"], s))
    if srv:
        lines.append(
            "  server steps in window (+-1 s): scen rep len | steps mixed/pre/dec"
            " | prefill tok | max step ms | mean ms mixed/dec | steps w/o all decoders"
        )
        for scen, rep, n, s in srv:
            lines.append(
                f"    {scen:<8} r{rep} {n:>6} | {s['steps']:>4} {s['steps_mixed']}/"
                f"{s['steps_prefill_only']}/{s['steps_decode_only']} | "
                f"{s['prefill_tokens']:>6} | {fmt(s['max_step_ms'])} | "
                f"{fmt(s['mean_step_ms_mixed'])}/"
                f"{fmt(s['mean_step_ms_decode_only'])} | "
                f"{s['steps_missing_decoders']}"
            )
    warn = []
    for a in result["runs"]:
        if a.get("error"):
            warn.append(f"{a['scenario']} r{a['repeat']}: {a['error']}")
        for e in a.get("stream_errors", []):
            warn.append(f"{a['scenario']} r{a['repeat']}: {e}")
        if a.get("decoders_finished_early"):
            warn.append(
                f"{a['scenario']} r{a['repeat']}: decoders hit max_tokens "
                f"before the scenario ended: {a['decoders_finished_early']}"
            )
    for w in warn:
        lines.append(f"  WARN {w}")
    gates = result["gates"]
    if gates:
        for g in gates:
            lines.append(
                f"  gate {g['gate']:<34} {g['window']:<20} {fmt(g['value'], 2):>10} "
                f"vs {g['limit']:<8} {'ok' if g['ok'] else 'FAIL'}"
            )
    lines.append(f"verdict: {result['verdict']}")
    return "\n".join(lines)


def write_timeline(path: str, runs: list[Run], clock: Clock) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "scenario",
                "repeat",
                "stream",
                "role",
                "kind",
                "t_s",
                "wall_s",
                "tokens",
                "cum_tokens",
                "gap_ms",
                "info",
            ]
        )
        for run in runs:
            for name, t, info in run.events:
                w.writerow(
                    [
                        run.scenario,
                        run.repeat,
                        "",
                        "event",
                        name,
                        f"{t:.6f}",
                        f"{clock.wall(t):.6f}",
                        "",
                        "",
                        "",
                        info,
                    ]
                )
            for s in run.streams:
                w.writerow(
                    [
                        run.scenario,
                        run.repeat,
                        s.sid,
                        s.role,
                        "submit",
                        f"{s.submit:.6f}",
                        f"{clock.wall(s.submit):.6f}",
                        "",
                        "",
                        "",
                        s.prompt_tokens,
                    ]
                )
                cum, prev = 0, s.submit
                for t, n in s.chunks:
                    cum += n
                    w.writerow(
                        [
                            run.scenario,
                            run.repeat,
                            s.sid,
                            s.role,
                            "token",
                            f"{t:.6f}",
                            f"{clock.wall(t):.6f}",
                            n,
                            cum,
                            f"{(t - prev) * 1000:.3f}",
                            "",
                        ]
                    )
                    prev = t


def _clean(o: Any) -> Any:
    """NaN -> None so the JSON is strict."""
    if isinstance(o, float):
        return None if math.isnan(o) or math.isinf(o) else o
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    return o


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--base-url", required=True, help="e.g. http://127.0.0.1:18120")
    ap.add_argument("--model", required=True, help="served model name")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--scenarios", default="solo,inject,periodic,reverse")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument(
        "--salt",
        default=None,
        help="prompt salt (default: per-invocation, defeats prefix cache)",
    )
    ap.add_argument("--corpus-file", default=None, help="text to tokenize for prompts")
    # Decoders.
    ap.add_argument("--decoders", type=int, default=8)
    ap.add_argument("--decoder-prompt-tokens", type=int, default=128)
    ap.add_argument("--decoder-max-tokens", type=int, default=8192)
    ap.add_argument("--decoder-stagger-s", type=float, default=0.0)
    ap.add_argument(
        "--warmup-tokens",
        type=int,
        default=16,
        help="tokens every decoder must have before 'steady'",
    )
    ap.add_argument("--steady-timeout-s", type=float, default=600)
    ap.add_argument("--baseline-s", type=float, default=8.0)
    # Injections.
    ap.add_argument("--prefill-lens", type=int_list, default=[4096, 16384])
    ap.add_argument("--inject-max-tokens", type=int, default=4)
    ap.add_argument("--recover-s", type=float, default=5.0)
    ap.add_argument("--periodic-len", type=int, default=4096)
    ap.add_argument("--period-s", type=float, default=6.0)
    ap.add_argument("--period-count", type=int, default=4)
    ap.add_argument(
        "--reverse-len",
        type=int,
        default=None,
        help="default: largest of --prefill-lens",
    )
    ap.add_argument("--reverse-delay-s", type=float, default=1.0)
    ap.add_argument("--settle-s", type=float, default=2.0)
    ap.add_argument("--request-timeout-s", type=float, default=1800)
    # Analysis.
    ap.add_argument(
        "--stall-factor",
        type=float,
        default=5.0,
        help="gap > factor x baseline ITL p50 counts as stalled",
    )
    ap.add_argument("--stall-min-ms", type=float, default=0.0)
    ap.add_argument(
        "--server-log",
        default=None,
        help="serve log with --enable-logging-iteration-details lines",
    )
    ap.add_argument("--timeline", action="store_true", help="write timeline.csv")
    # Gates.
    ap.add_argument("--max-stall-ms", type=float, default=None)
    ap.add_argument("--max-stalled-s", type=float, default=None)
    ap.add_argument(
        "--min-decode-tokens-during-prefill",
        type=float,
        default=None,
        help="per-decoder minimum tokens during each inject window",
    )
    ap.add_argument("--max-ttft-slowdown", type=float, default=None)
    ap.add_argument("--max-starved", type=float, default=None)
    ap.add_argument(
        "--gate-stat",
        choices=["median", "worst"],
        default="median",
        help="statistic across repeats compared against gates",
    )
    ap.add_argument(
        "--gate-reverse",
        action="store_true",
        help="also gate the reverse scenario windows",
    )
    a = ap.parse_args(argv)
    a.scenarios = [s.strip() for s in a.scenarios.split(",") if s.strip()]
    bad = set(a.scenarios) - {"solo", "inject", "periodic", "reverse"}
    if bad:
        ap.error(f"unknown scenarios {sorted(bad)}")
    if a.reverse_len is None:
        a.reverse_len = max(a.prefill_lens)
    if a.salt is None:
        a.salt = str(time.time_ns())
    return a


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    client = Client(args.base_url, args.model, args.request_timeout_s)
    corpus = CORPUS
    if args.corpus_file:
        with open(args.corpus_file) as f:
            corpus = f.read()
    pool, source = build_pool(client, corpus, args.seed)
    probe = Probe(args, client, Prompts(pool, args.seed, args.salt))

    runs: list[Run] = []
    t_start = time.time()
    solo_needed = {args.periodic_len, args.reverse_len, *args.prefill_lens}
    for scen in args.scenarios:
        for r in range(args.repeats):
            if scen == "solo":
                saved = args.prefill_lens
                args.prefill_lens = sorted(solo_needed)
                run = probe.solo(r)
                args.prefill_lens = saved
            else:
                run = getattr(probe, scen)(r)
            runs.append(run)
            print(
                f"[stall_probe] {scen} r{r} done"
                f"{' ERROR ' + run.error if run.error else ''}",
                flush=True,
            )
            probe.wait_idle()

    solo: dict[int, list[float]] = {}
    for run in runs:
        if run.scenario == "solo":
            for s in run.injects:
                if not math.isnan(s.ttft):
                    solo.setdefault(s.prompt_tokens, []).append(s.ttft)
    solo_ttft = {k: statistics.median(v) for k, v in solo.items()}
    dec_ttfts = [
        s.ttft
        for run in runs
        if run.scenario in ("inject", "periodic")
        for s in run.decoders
        if not math.isnan(s.ttft)
    ]
    ref_dec_ttft = statistics.median(dec_ttfts) if dec_ttfts else math.nan

    iters: list[Iteration] = []
    if args.server_log:
        iters = parse_iteration_log(
            args.server_log, dt.datetime.fromtimestamp(probe.clock.wall0).year
        )
    analyses = [
        analyze_run(run, args, solo_ttft, ref_dec_ttft, iters, probe.clock.wall0)
        for run in runs
    ]
    agg = aggregate(analyses)
    gates = evaluate_gates(agg, args)
    errors = any(a.get("error") or a.get("stream_errors") for a in analyses)
    if errors:
        verdict = "ERROR"
    elif not gates:
        verdict = "NO-GATES"
    else:
        verdict = "PASS" if all(g["ok"] for g in gates) else "FAIL"
    cfg = {k: v for k, v in vars(args).items()}
    result = {
        "config": cfg,
        "prompt_source": source,
        "started_wall": t_start,
        "clock_wall0": probe.clock.wall0,
        "server_log_iterations": len(iters),
        "runs": analyses,
        "aggregate": agg,
        "gates": gates,
        "verdict": verdict,
    }
    with open(os.path.join(args.out, "stall_probe.json"), "w") as f:
        json.dump(_clean(result), f, indent=1)
    text = summary_text(result)
    with open(os.path.join(args.out, "summary.txt"), "w") as f:
        f.write(text + "\n")
    if args.timeline:
        write_timeline(os.path.join(args.out, "timeline.csv"), runs, probe.clock)
    print(text)
    if verdict == "ERROR":
        return 2
    return 1 if verdict == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main())
