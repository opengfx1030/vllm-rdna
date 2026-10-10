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

## Baseline measurements (2026-10-09)

Setup: `par1-cs25`, 4× V620 TP=4 on HIP 6-9, FULL_AND_PIECEWISE, V2 runner,
prefix caching on, `SEQS=8`, `MAXBAT=2048`. v0.31 is the `vllm-rdna-0.31.0-d`
tree (port tip as of 2026-10-08) with venv `venv-7.14.0_0.31.0-d`; 0.28 is
`vllm-rdna-0.28.0` with the PLE int4 sidecar. Driver: `stall_ab.sh` with
`STALL_ARGS="--decoders 6 --prefill-lens 4096,16384 --periodic-len 4096
--period-s 6 --period-count 4 --reverse-len 16384 --repeats 3 --baseline-s 8
--recover-s 5"`. Raw results are in
`~/w4a8_runs/port-v031/serve-f-{fn,27b,fn028}-<arm>/stall/` and
`stall-ab-f-*.txt` on the box.

The tables give medians over 3 repeats; the sd column of the summaries is
≤ 2 % except where noted. Baseline decoder ITL p50 is 33.7-35.0 ms in every
arm. "x solo" is the injected request's TTFT divided by its solo TTFT.

### Flash-Next MTP0 (`flashnext-mtp0`), 16k injection into 6 decoders

| arm | max gap ms | gap p50 ms | frozen | dec tok (min/dec) | dec rate vs base | TTFT s (x solo) | reverse: decoders blocked s (tok) |
|---|---:|---:|---:|---:|---:|---:|---:|
| v0.31 default | 515 | 508 | 0.99 | 114 (19) | 0.08 | 8.25 (1.11) | 7.13 (0) |
| v0.31 `PREFILL_INTERVAL=4` | 517 | 34 | 0.82 | 419 (69) | 0.24 | 9.97 (1.35) | 7.18 (0) |
| v0.31 `LPTH=256` | 252 | 247 | 0.99 | 404 (67) | 0.14 | 15.95 (2.13) | 2.69 (236) |
| v0.31 interval 4 + LPTH 256 | 247 | 34 | 0.70 | 1560 (260) | 0.40 | 22.21 (3.01) | 2.66 (921) |
| 0.28 default | 547 | 522 | 0.99 | 112 (18) | 0.08 | 8.53 (1.10) | 7.00 (10) |
| 0.28 interval 4 + LPTH 256 | 287 | 35 | 0.72 | 1531 (255) | 0.37 | 24.19 (1.57) | 0.96 (1357) |
| v0.31 default, rerun (drift) | 505 | 497 | 0.98 | 119 (19) | 0.08 | 8.10 (1.10) | (server died) |

4k injection, in the order of the first five rows: max gap 504 / 511 / 246 /
241 / 524 ms; decode tokens during the prefill 40 / 126 / 117 / 402 / 34;
x solo 1.14 / 1.40 / 2.16 / 3.04 / 1.09. The rerun of the v0.31 default arm
two hours later reproduces the first run within 2-4 % (505 vs 515 ms, 119 vs
114 tokens) before rdna_ar wedged in its periodic scenario. On 0.28 the
combination costs less prefill time (x solo 1.57 vs 3.01): 0.28's solo TTFT
under `LPTH=256` is already 15.4 s, against 7.4 s on v0.31, because the cap
also applies to the solo prefill there. On v0.31 upstream skips the cap for
a sole request, so x solo grows instead. The periodic 4k slots match the 4k inject row in every
arm (for example default: 510-513 ms max gap, 40-48 tokens).

### Qwen3.8-27B AWQ (`27b-awq`), 16k injection into 6 decoders

| arm | max gap ms | gap p50 ms | frozen | dec tok (min/dec) | dec rate vs base | TTFT s (x solo) | reverse: decoders blocked s (tok) |
|---|---:|---:|---:|---:|---:|---:|---:|
| v0.31 default | 2672 | 1978 | 0.99 | 70 (11) | 0.02 | 18.38 (1.03) | 17.84 (0) |
| v0.31 interval 4 + LPTH 256 | 396 | 36 | 0.75 | 1548 (258) | 0.33 | 27.54 (1.51) | 3.80 (1129) |

4k injection: default max gap 1972 ms, 32 tokens, x solo 1.06; combo 296 ms,
387 tokens, x solo 1.61.

### What the probe shows

- **The stall reproduces in every run and does not come from the port.**
  With default settings, every decoder is frozen for 95-99 % of each
  prefill. Flash-Next decoders get one token per ~510 ms mixed step against
  a 34 ms baseline, about 0.08x their normal rate. The 27B gets one token per
  2.0-2.7 s step (0.02x). 0.28 behaves the same as v0.31 (Flash-Next
  524-547 ms, 0.08x).
- **Reverse direction.** Decoders that arrive during a 16k prefill get
  nothing until the prefill ends: 7.1 s on Flash-Next and 17.8 s on the 27B.
  The running prefill takes the whole 2048-token budget and new requests
  cannot be admitted. The cadence knob cannot help here, because no decoder
  is running yet. `LPTH` does help (2.7 s / 3.8 s).
- **Step composition** (server log, Flash-Next v0.31). A pure decode step
  takes 33 ms. A mixed step with 1024 prompt tokens takes ~475 ms (4k
  prompt: ~400 ms), one with 256 prompt tokens ~240 ms, and a prefill-only
  step with 2048 tokens ~900 ms. **Any step that carries prefill pays about
  200 ms on top of decode, whatever the chunk size.** That floor sets the
  stall length once chunks are small. In mixed steps the prefill chunk is
  1024 tokens, not 2048: the 2048 budget minus the decode tokens is rounded
  down to the 1024-token mamba block (`mamba-align`). Solo prefill runs
  2048-token chunks.
- **Knob sensitivity.** The probe separates the knobs cleanly:
  - `PREFILL_INTERVAL=4` gives decoders pure-decode steps between
    prefill-carrying steps. Gap p50 drops to the baseline, decode tokens
    rise 3.7x and prefill TTFT rises 1.35x, but the max gap stays at
    ~515 ms (the release steps).
  - `LPTH=256` halves the max gap and fixes the reverse case, but prefill
    takes 2.1x as long and gap p50 stays at the ~240 ms floor.
  - The combination gives the most decode service: 0.40x the baseline rate,
    13.7x the tokens, max gap 247 ms. Prefill takes 3x as long (Flash-Next)
    or 1.5x (27B).
- **Rare outlier caught in passing.** In the Flash-Next `PREFILL_INTERVAL=4`
  arm, one mixed step (1024 prompt + 6 decode tokens) took 20.6 s in one
  repeat. There is no fault and no log line. KV usage crossed 14 → 15.7 % at
  that moment. It shows as max gap 511±11603 ms (the median is unaffected).
  Not followed up here.
- **Noise.** The rdna_ar one-shot all-reduce wedged in 3 of 8 Flash-Next
  runs on HIP 6-9 (`peer rank N's flag never arrived`): twice on the first
  request, once mid-run at collective #200598. It never wedged in the 27B
  runs or on 0.28. A wedged arm reports `verdict: ERROR` (exit 2) and the
  probe stops early. `stall_ab.sh` now clears the wedge marker between arms, because
  the marker otherwise silently forces RCCL on every later arm sharing the
  cache.

### Suggested gates for fix rounds

Use the Flash-Next MTP0 recipe, `--decoders 6 --prefill-lens 4096,16384
--repeats 3`, and `--gate-stat median`:

| gate | default today | target for a fix | rationale |
|---|---:|---:|---|
| `--max-stall-ms` | 515 (27B 2672) | 300 | below one 1024-token mixed step; the LPTH arm already reaches 250 |
| `--min-decode-tokens-during-prefill` (per decoder, 16k) | 19 (27B 11) | 100 | ~0.4x baseline service over a ~8 s prefill; the combo reaches 260 |
| `--max-starved` | 0 | 0 | any decoder at zero tokens during a prefill is a regression |
| `--max-ttft-slowdown` | 1.11 | 1.5 | keeps a fix from just starving the prefill (the combo is at 3.0 on Flash-Next) |
| `--gate-reverse --max-stall-ms` | 7130 | 3000 | blocked arrivals; LPTH reaches 2.7 s |

None of today's arms passes all of them; that is the point. A fix has to
cut the ~200 ms floor of a prefill-carrying step, or overlap prefill with
decode, rather than only trade decode ITL against TTFT.

## Decode-stall cap (scheduler, `vllm/v1/core/sched/mixed_step.py`)

While at least one request is decoding, the cap bounds the prefill tokens of a
step so that its predicted duration stays within a budget. The prediction is a
line `t = a + b * prefill_tokens` fitted to measured mixed steps. Optionally,
the cap also reserves a share of wall time for pure decode steps. A lone
prefill (no decoders) is never capped.

| Env | Default | Meaning |
|---|---|---|
| `VLLM_RDNA_DECODE_STALL_MS` | `0` (off) | step-duration budget in ms; 250 in the dense recipes |
| `VLLM_RDNA_DECODE_SHARE` | `0` | fraction of wall time owed to pure decode steps after a mixed step |
| `VLLM_DECODE_STALL_MAX_OVERHEAD` | `0.25` | efficiency floor: largest fixed-cost / step ratio the cap accepts before it gives up shrinking the chunk |
| `VLLM_DECODE_STALL_FIT_ALL` | auto | which steps feed the fit: unset = all prefill steps on dense models, mixed steps only on MoE; `1` / `0` force either |

The fit mode depends on the model (`MixedStepController.fit_all_steps_default`):

- **MoE (Flash-Next): mixed steps only.** Prefill-only steps (deep-context
  chunks, prompt-logprob steps) cost 330-795 ms of fixed time, which pushed the
  fit above the budget, so the cap went back to whole 1024-token chunks.
- **Dense (27B): all prefill-carrying steps.** With mixed steps only, the 27B
  fit drifted up on deep-context mixed chunks: in the final pass the 16k
  injection max gap was 472 ms on the first repeat, then 775 / 793 ms (mixed
  steps 690-710 ms), against 394 ms with the all-steps fit.

With one bucket of samples the cap can also drop below its initial 512 tokens.

### Where it is on

**Dense 27B (`27b-awq`, `27b-exl3`, `full`): on, budget 250 ms.** 16k injection
into 6 decoders, TP=4 F&P:

| Metric | off | cap 250 ms |
|---|---:|---:|
| Longest decode gap during a 16k prefill | 2672 ms | **394 ms** |
| Decode tokens per decoder during that prefill | 11 | 81 |
| Decoders blocked in the reverse case | 17.8 s | 4.8 s |
| 1k/512 c=8 TTFT | 3.88 s | 2.68 s |
| 16k/1k c=8 aggregate output | 41.2 tok/s | 37.5 tok/s (−9 %) |
| c=1 cells | — | unchanged |

**Flash-Next (`flashnext-mtp0`, `flashnext-mtp2`): opt-in.** The trade-off is
worse than on the dense model. Flash-Next mixed steps are GPU-bound, at about
95 ms + 0.40 ms per token, so shorter stalls need smaller chunks, and smaller
chunks cost prefill and c=8 throughput. MTP0 measurements (16k injection into
6 decoders, fused HC prefill on, capture sizes up to 1032 via `CG_SIZES`):

| Arm | Longest gap | Decode tok (worst decoder) | Prefill TTFT × solo | Reverse blocked | Mixed chunk | 16k/1k c=8 agg. output | 16k/1k c=8 TTFT |
|---|---:|---:|---:|---:|---:|---:|---:|
| off | 505 ms | 115 (19) | 1.13 | 6.8 s | 1024 | 85.1 tok/s | 15.9 s |
| budget 280 ms, floor 0.6, old fit | 528 ms | 111 (18) | 1.18 | 3.1 s | 1024 | — | — |
| budget 250 ms, floor 1.0, old fit | 516 ms | 115 (19) | 1.16 | 3.0 s | 1024 | — | — |
| budget 250 ms, floor 1.0, mixed-step fit | **341 ms** | **215 (35)** | 1.52 | 3.1 s | 512 | 74.1 tok/s (−13 %) | 19.2 s |

All other cells moved by about 3 % or less. All correctness gates passed in
every arm.

To turn it on for Flash-Next, add these to the serve command line:

```bash
bash tools/rdna/serve_rdna.sh RECIPE=flashnext-mtp0 \
  VLLM_RDNA_DECODE_STALL_MS=250 VLLM_DECODE_STALL_MAX_OVERHEAD=1.0 \
  CG_SIZES='[1,2,4,8,16,32,64,128,256,384,512,768,1024,1032]'
```

Use it when interactive latency for requests that are already decoding matters
more than 16k c=8 throughput, for example a chat front end with long pasted
contexts.

The ≤ 300 ms gate is not reached yet: the cap stayed at its initial 512
tokens. The single-bucket fix should give ~384-token chunks and ~280 ms steps,
but it is not measured on GPU yet. Run `tools/rdna/port_v031/ladder_warmup.py`
(called by `serve_validate.sh`; `WARMUP=0` skips it) before measuring with a
capture ladder. Otherwise the first use of each new size compiles in the middle
of the bench.
