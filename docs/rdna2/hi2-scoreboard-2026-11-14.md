# HI2 27B TP=2 scoreboard: one driver, one JSON and one markdown per arm

Every change to HI2 inference (a fused kernel, a flag, a recipe edit) is judged by the same
scoreboard. One **arm** is one boot of `RECIPE=27b-hi2-tp2` with a set of env switches or
recipe overrides; `tools/rdna/rdna_hint_flex/scoreboard.sh ARM [KEY=value ...]` boots it
once, runs everything below against that server and leaves `scoreboard.json` and
`scoreboard.md` in `$OUT_ROOT/ARM/` (default `~/w4a8_runs/scoreboard/ARM/`).
Labels: **[measured]** = a run listed here, **[inference]** = a run must confirm.

## What one arm records

1. **Per request**, for every bench cell: TTFT, prefill tok/s (prompt tokens / TTFT),
   decode tok/s (output tokens after the first / time from first to last token), ITL
   p50 and p99 (pooled over the streamed gaps; `--save-detailed` gives the per-request lists).
2. **Aggregate** per cell: output tok/s, total tok/s, request/s and goodput (output tokens
   of requests with TTFT <= 2 s + 1 s per prompt ktoken and ITL p99 <= 100 ms, per second of
   cell wall time; the SLO flags are on `scoreboard_report.py arm`, compare goodput only
   between arms judged with the same SLO). Default cells (`CELLS=`, input:output:conc:requests):
   `1024:512:1:16 1024:512:8:16 16384:1024:1:8 16384:1024:8:16 32768:2048:1:4`.
   The prefix cache is cleared before every cell (`RESET_PREFIX_CACHE=1`, needs
   `VLLM_SERVER_DEV_MODE=1`, both set by the driver): the cells share one random-prompt seed,
   so a later cell would otherwise be served from the cache of an earlier one (the 8k and
   16k cells of an earlier A/B shared their first 8192 tokens that way).
3. **Quality**, all collected on the live server after the cells (`scoreboard_hook.sh`, the
   existing `tools/rdna/port_v031` tools unchanged): short PPL (the serve_validate probe),
   long-context PPL (`quality_ab.py`, 6 passages of about 15k tokens, plus 96 greedy
   tokens each), prompt-level top-k KL, dNLL and top-1 (`topk_kl.py`, 64 windows of 2048
   tokens, one prefill chunk each) and decode-path KL (`decode_kl.py`, 32 passages x 512
   greedy tokens, 8 concurrent). The text is cut from a corpus frozen at
   `$OUT_ROOT/corpus.txt` (the docs of the tree that first ran), so every arm sees the same
   tokens. Against `BASELINE=` the report prints each statistic for the arm and for
   `FLOOR=` (the baseline booted a second time, same flags): a delta is a finding only if
   it exceeds the floor (`judged` rows: arm within 1.25x of the floor, or above).
4. **Validity**: greedy probe before and after load, PPL probe, prefix-cache probe,
   concurrency probe, KFD queue-eviction time per cell (a non-zero cell is INVALID),
   amdgpu fault markers, junction / edge / memory temperature and VRAM at 1 s
   (`thermal.csv`; worst rolling 5 s mean against 90 C), speculative acceptance at c=1 and
   c=4 when the arm has `MTP=` > 0, and the **window label**: "quiet protocol" when
   `QUIET_HOLD=1` (default: the driver waits for any `docker buildx build` and for other
   lanes' windows) and otherwise "not for comparison".

## Running arms

The recipes `27b-hi2-tp2` and `27b-hi2-tp2-mtp2` turn the HI2 switches on
([`hi2-27b-tp2-serving.md`](hi2-27b-tp2-serving.md)), so an arm that means "flags off" names
the opt-outs: `OFF` below is that list, and a later `KEY=value` wins over an earlier one.

```bash
# on the box, my lane (HIP 2,3 = PCI d9 dc; the driver refuses if HIP != BUSES)
OFF="RDNA_HINT_FLEX_GEMV=v1 VLLM_FA_RDNA2_UNFUSED_PREFILL=0 VLLM_RDNA_GEMMA_RESIDUAL_DIRECT=0
 VLLM_RDNA_GDN_DECODE_INPLACE=0 RDNA_HINT_FLEX_FUSE_SILU=0 VLLM_RDNA_LM_HEAD_GEMV_MAX_M=8
 TUNABLEOP_PREFER="
scoreboard.sh base $OFF                                # flags off: the baseline
scoreboard.sh base2 $OFF                               # second boot of the baseline: the floor
BASELINE=base FLOOR=base2 scoreboard.sh attn $OFF VLLM_FA_RDNA2_UNFUSED_PREFILL=1
BASELINE=base FLOOR=base2 scoreboard.sh mtp2 $OFF MTP=2
python tools/rdna/rdna_hint_flex/scoreboard_report.py table \
    ~/w4a8_runs/scoreboard/{base,base2,attn,mtp2} --out table.md   # one row per arm and cell
```

Variables: `OUT_ROOT GPUS BUSES PORT RECIPE MODEL VENV SERVE_TREE CELLS QUALITY QUIET_HOLD
KEEP_CACHE TREE_SHA` (the box tree has no `.git`; pass the sha it was synced from).
`SB_SMOKE=1` shrinks the quality passes for a harness check; give it its own `OUT_ROOT`
(corpus and windows are frozen on first use). Run the report alone on stored results with
`scoreboard_report.py arm DIR --baseline DIR --floor DIR --python <venv python>` (the
top-k comparison needs numpy).

## Evidence that the harness works [measured]

A smoke arm (`OUT_ROOT=~/w4a8_runs/scoreboard-smoke SB_SMOKE=1 CELLS=1024:64:1:2
QUIET_HOLD=0 scoreboard.sh smoke`, HIP 2,3, 2026-11-15 01:25 box time) booted in 191 s, passed the
greedy, PPL, prefix and concurrency probes, ran the cell (TTFT 1.47 s, decode 36.8 tok/s per
request, evictions +0/+0), ran the three quality passes at smoke size (about one minute) and wrote the
report with thermals (junction 78 / 74 C) and VRAM (27.1 GiB). The baseline / floor plumbing
was exercised by comparing the smoke arm with itself (every delta zero, "within floor"):
that checks the code path, not a result. The noise floor of the real quality passes was
measured by `base2` against `base`: long-context PPL 3.6633 both, 6/6 identical greedy
continuations, prompt KL 0 (top-1 1.0000), decode KL 2.37e-6 nats with 28/32 identical
decodes, output tok/s within 0.1% in every cell. A full arm takes about 45 to 50 minutes
[inference: 3 min boot, 5 min probes, about 21 min of cells, a quality pass, quiet gaps].

## Queue [done]

`~/w4a8_runs/hi2-tp2/sb_queue.sh` ran `base`, `base2` (floor) and `attn`
(`VLLM_FA_RDNA2_UNFUSED_PREFILL=1`, baseline `base`, floor `base2`) one after the other on
HIP 2,3 under the quiet protocol, starting 2026-11-15 01:33 box time; the results are in
`~/w4a8_runs/scoreboard/ARM/scoreboard.md`. Later arms (`gemv`, `residual`, `gdnglue`,
`stack`, `tunable`, `final`, the MTP arms and the concurrency pairs) used the same driver:
the arm list, tables and tree SHAs are in [`hi2-27b-tp2-serving.md`](hi2-27b-tp2-serving.md),
copies of the result files in `bench_results/2026-11-15_hi2-scoreboard/`.

## Limits

The corpus is this repository's docs (English technical text), so quality statistics
describe that text only. The windows are one chunk each (the prefill kernels of a 2048-token
chunk, not of a long prompt: the long-context PPL covers those). Speculative arms are
compared to an MTP 0 baseline on the same target model, so their decode KL should sit near
the floor [inference]. The hook does not open its own quiet window (the quality passes are
correctness runs, not timings).
