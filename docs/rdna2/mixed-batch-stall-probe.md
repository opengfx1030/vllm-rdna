# Mixed-batch stall probe

`tools/rdna/port_v031/stall_probe.py` catches, measures and reproduces the
mixed prefill/decode stalls seen on the 0.28 fork. While one request
prefills a long prompt, the requests already decoding freeze, and sometimes
the reverse happens. It is a measuring tool only. It changes nothing in the
scheduler or the kernels. Use it to A/B fix rounds.

Earlier work on the same symptom: `prefill-cadence-2026-09-28.md`,
`prefill-decode-interference-2026-09-27.md` and
`mixed-batch-and-prefill-investigation-2026-10-05.md` in the parent repo.
Those used `tools/rdna2_028/prefill_decode_probe.py` (one decoder plus one
16k injection: 9 decoder tokens during a 9.6 s prefill, ITL p50 1.5 s) and
`bench_results/2026-10-05_mixedbatch-validation/mixed_probe.py` (7 decoders
plus one 16k prompt, which proved that mixed steps happen:
`Running: 8, Waiting: 0`). This probe generalizes their inject scenario.
It runs N decoders, several lengths, periodic and reverse cases and repeats,
uses token-exact prompts, adds gates and attributes each window to scheduler
steps. Fork knobs that move it (all still on v0.31):

- `--prefill-schedule-interval` (`PREFILL_INTERVAL=`): base `EngineCore`
  cadence.
- `--long-prefill-token-threshold` (`LPTH=`): per-step chunk cap. The
  scheduler skips the cap when only one request is present. Upstream's
  adaptive LPT (`--long-prefill-token-threshold-adaptive`, off by default)
  floors it at `max_num_batched_tokens / num_requests`.
- `VLLM_RDNA_DYNAMIC_PREFILL=1` + `VLLM_RDNA_DYN_*`
  (`vllm/v1/core/sched/dynamic_prefill.py`): live tuner for both knobs.

## Why `vllm bench serve` misses this

`--request-rate inf --max-concurrency N` submits every prompt at once. The
engine then runs prefill steps first and decode steps after, almost never in
the same step. A mixed step only happens when a long prompt arrives while
other requests are already decoding. The probe builds exactly that case, at
fixed times.

## What it does

The probe drives a live server through streaming `/v1/completions` (stdlib
only, no tokenizer needed locally). Prompts are token-id lists of an exact
length, built from a fixed corpus that the server's `/tokenize` encodes once.
Each request gets its own 16-token header derived from `--seed`, `--salt`,
the scenario, the repeat and the stream, so prefix caching never hides
prefill work. Token counts come from `continuous_usage_stats`, which stays
exact with MTP's multi-token chunks. Timestamps use `time.monotonic()`.

Scenarios (`--scenarios`, run in this order, each `--repeats` times):

| scenario | what happens | what it exposes |
|---|---|---|
| `solo` | every prefill length alone | reference TTFT per length |
| `inject` | N decoders reach steady decode (every one has `--warmup-tokens`), then a `--baseline-s` window, then each `--prefill-lens` prompt is injected one at a time. After each one the probe waits for its first token plus `--recover-s`. | decoder freeze during a one-shot long prefill |
| `periodic` | N steady decoders. One `--periodic-len` prompt every `--period-s`, `--period-count` times; the prompts may overlap. | repeated or overlapping prefills, recovery between them |
| `reverse` | one `--reverse-len` prompt first, then N decoders `--reverse-delay-s` later | new decoders blocked behind a running prefill (their TTFT), and the prefill slowed by them |

Injection times are fixed offsets from the steady-state moment, so the runs
can be compared.

## Metrics (per injection window = prefill submit to its first token)

- **gaps p50/p90/p99/max**: decoder inter-token gaps that overlap the window,
  pooled over decoders, next to the baseline window's gaps.
- **max gap**: the longest stall.
- **stalled s**: per decoder, the sum of gaps longer than the threshold
  (`--stall-factor` × baseline ITL p50, at least `--stall-min-ms`). The
  report gives the worst decoder.
- **frz**: frozen fraction of the window, i.e. stalled time clipped to the
  window divided by its length (worst decoder). 1.0 means the decoder
  produced nothing during the whole prefill.
- **dec tok**, **min/dec**: decode tokens produced while the prefill was in
  flight, in total and for the worst decoder.
- **retain**: decode rate in the window divided by the baseline decode rate.
- **jain**, **starv**: fairness. Jain index of per-decoder tokens in the
  window (1 means even), and the number of decoders that got zero tokens.
- **TTFT s**, **x solo**: the injected request's TTFT, and that TTFT divided
  by its solo TTFT (prefill slowdown).
- `reverse` also reports the decoders' TTFT (`decoder_ttft_s` in the JSON)
  against the TTFT they get without a running prefill
  (`decoder_ttft_ref_s`), and `decoders_served_during_prefill`. Its gap
  metrics count the wait for the first token.
- `periodic:busy`: the same metrics over the whole span from the first
  submit to the last first token.

The summary prints median±sd across repeats. `stall_probe.json` holds every
window, the per-decoder numbers and spreads (mean/sd/min/max/median).

### Server-side corroboration (optional, read-only)

Start the server with upstream's `--enable-logging-iteration-details` and
pass its log with `--server-log`. The flag only adds log lines. Each window
is then annotated with the scheduler steps that ran in it: mixed,
prefill-only and decode-only steps, prompt tokens per step, max and mean step
time, and `steps w/o all decoders` (steps that left some decoders out, e.g.
because the batch was full or the cadence deferred them). Log stamps have
1 s resolution, so the attribution is ±1 s.

## Running

Against a running server:

```bash
python tools/rdna/port_v031/stall_probe.py --base-url http://127.0.0.1:18271 \
    --model flash-next --out ~/w4a8_runs/port-v031/f-stall-x \
    --decoders 6 --prefill-lens 4096,16384 --reverse-len 16384 --repeats 3 \
    --server-log ~/w4a8_runs/port-v031/serve-x/serve.log --timeline
```

Keep `decoders + overlapping injections <= --max-num-seqs`; the recipes use
`SEQS=8`, so 6 decoders leave room for two overlapping periodic prompts.
Otherwise the injected request queues instead of mixing. Keep
`--decoder-max-tokens` (default 8192) above what a scenario consumes; the
summary warns when a decoder finished early.

Inside a validation run (boots the recipe, runs the greedy probes, then the
stall probe; `CELLS=none` skips the bench cells):

```bash
STALL_PROBE=1 CELLS=none STALL_ARGS="--decoders 6 --max-stall-ms 1500" \
RECIPE=flashnext-mtp0 MODEL=... GPUS=6,7,8,9 TAG=x \
    bash tools/rdna/port_v031/serve_validate.sh [PREFILL_INTERVAL=4 LPTH=256 ...]
```

`STALL_PROBE=1` also adds `--enable-logging-iteration-details` to the server
(`STALL_ITER_LOG=0` turns that off). A trailing `EXTRA_ARGS=...` override
replaces it.

Knob A/B, one boot per arm, one shared compile cache:

```bash
RECIPE=flashnext-mtp0 MODEL=... GPUS=6,7,8,9 TAG=fn-stall \
ARMS="base PREFILL_INTERVAL=4 LPTH=256 PREFILL_INTERVAL=4,LPTH=256" \
LOCK=~/w4a8_runs/port-v031/GPU69_LOCK STALL_ARGS="--decoders 6" \
    bash tools/rdna/port_v031/stall_ab.sh
```

The combined report goes to `~/w4a8_runs/port-v031/stall-ab-$TAG.txt`.

### Determinism

- `--salt` defaults to a new value per invocation, so a second run against
  the same server never hits the first run's prefix cache. Pin it
  (`--salt 1`) on a freshly booted server, or with prefix caching off, for
  bitwise-identical prompts across invocations.
- Greedy decoding (`temperature 0`) and `ignore_eos`. Output text is not
  compared, only timing.
- Variance: read the ±sd column. On the V620 box, 3 repeats were enough to
  separate the arms below.

### Gates (regression use)

Gates are off unless set. `--gate-stat median` (the default) compares the
median across repeats; `worst` compares the worst repeat. Exit code: 0 pass
or no gates, 1 a gate failed, 2 request errors or no steady state.

| flag | checks | windows |
|---|---|---|
| `--max-stall-ms` | max decoder gap | inject, periodic (+ reverse with `--gate-reverse`) |
| `--max-stalled-s` | worst decoder's stalled time | same |
| `--min-decode-tokens-during-prefill` | min tokens per decoder during a prefill | inject, periodic slots |
| `--max-ttft-slowdown` | injected TTFT / solo TTFT | inject, periodic slots |
| `--max-starved` | decoders with zero tokens during a prefill | same |

## Reading the output

- `frz` near 1.0 with `dec tok` near 0: decoders froze for the whole
  prefill. That is the stall.
- A max gap of about one step of `MAXBAT` prompt tokens: mixed steps are
  slow, and every decoder in the step waits for the chunk. `LPTH` bounds it.
- High p99 but low p50: the cadence works, but the release steps still carry
  a full chunk.
- `x solo` well above 1: the prefill is the one being starved, e.g. by a
  cadence that defers it.
- `reverse` decoder TTFT far above its reference: arriving decoders wait
  behind the running prefill.
- In the server table, mixed steps with a large `max step ms` explain a long
  max gap. Steps without all decoders mean decoders were left out of steps.

## Baselines

See "Baseline measurements" below (v0.31 `-d` tree, 4× V620 TP=4,
FULL_AND_PIECEWISE).
