# T10: expert statistics on the production server (V620 slot B), real skew and plans

Test T10 of the hetero-MoE spec (`docs/rdna2/hetero-moe-design-2026-11-07.md`,
branch `feat/hetero-moe`): the routing collector of
`docs/rdna2/expert-stats-2026-11-07.md` on the production `flashnext-mtp2` TP4 + EP4
server, GPU validation items 1-8 of that document's section 8, about one hour of
agentic traffic, and the H1 / H2 placement plans solved from the real statistics
and simulated on a held-out window. Box `par1-cs25`, V620 slot B (HIP 6-9, buses
0a/0f/14/17), compiled mode 3, `FULL_AND_PIECEWISE`, no `--monokernel`, MAXBAT 2048,
all under the slot B lease. Sections 1-5 are measured on slot B; section 6 is the
follow-up run (T2) on HIP 2-5 of the interim box layout with the final-snapshot fix,
the fused kernels and the id trace.

## 1. Result in one table

| Item | Verdict | Evidence |
|---|---|---|
| 1 registration guard never fires | pass | B1 and B2 (collector on, FULL_AND_PIECEWISE captured after the warm-up): no guard `RuntimeError` in `serve.log`, 49 layers (48 target + MTP) registered on every rank |
| 2 counts == tokens x top_k, tokens == server | pass | per layer `sum(counts) == top_k * tokens`; target rows `final - snapshot 1` against the server's scheduled rows: B2 -0.07 % (the last tick precedes the post-load probe), B1 +0.16 % median over 153 snapshots (10 s poll interpolation, section 3) |
| 3 `index_add_` under graph replay | pass (by invariants) | decode counts against the independent row histogram: 0 of 49 layers off over 8.9 M rows (B1) and 0.7 M (B2); no counter ever decreased in 154 snapshots. The literal eager-vs-graph comparison of one prompt was not run (the two invariants plus the server row count are stricter over real traffic) |
| 4 rank agreement | pass | all four EP ranks record (see section 3): equal step counts, per-expert counts differ by at most 2 rows in 10^6-10^7 (<= 2.2e-6) |
| 5 overhead | torch formulation **fails the budget**; fused kernels pass | torch: +5.5 to +6.1 ms per step, +24.7 % ITL at c1, +13 to +14.5 % at c6/c8 (ABBA, resolved far beyond the spread). Fused Triton kernels (section 6, ABBA on HIP 2-5): +0.49 ms (+2.2 %) at c1, +0.37 ms (+1.6 %) at 16k c1, unresolved at c6/c8 (<= 0.9-2.4 ms) |
| 6 snapshot ordering | pass | `seq` +1, `elapsed_s` and every cumulative counter non-decreasing over 154 snapshots (B1) |
| 7 SIGTERM writes a final snapshot | **bug found, fixed, validated** | `close()` raised `RuntimeError: Inplace update to inference tensor outside InferenceMode` at shutdown, so the final snapshot was not written (the last periodic one, up to 30 s old, remains and parses). Fix: `close()` runs under `torch.inference_mode()`; a CPU test reproduces the error. On the box (section 6, arms F1, F2, C, G) every rank's last snapshot is newer than its last periodic one (16 of 16 files) |
| 8 stats -> solve -> plan -> held-out hub hit | pass | H1 held-out hub hit 0.580 and 0.583 against 0.586 and 0.582 in-sample; H2 0.784 and 0.786 against 0.789 and 0.788 (section 5; simulated with the correlated row model, 0.3 points below the analytic hit) |
| 9 row independence (follow-up) | **fails; model replaced** | the top-k id trace of arm G (section 6): the rows of a decode step share more experts than independent draws (3-9x within an MTP group); 11-34 % fewer distinct experts per 24-row step than independent rows. `simulate` and `solve` now draw correlated rows by default (section 6.1, fitted on that trace and checked against it: distinct experts within 1-6 %, step time within 3 %); `runtime.spill_k` and the step times below come from it |

Real skew (decode rows, mean over the 48 target layers): the top 8, 32, 64, 128 and
256 experts of a layer carry 9.9, 25.1, 38.8, 58.5 and 83.1 % of the routed rows
(uniform: 1.6, 6.3, 12.5, 25, 50 %), both windows within 0.5 points of each other.
The skew is real but moderate; the placement gain is a hub hit of 0.58 (H1) and 0.79
(H2) against 0.25 for a blind stripe.

## 2. Setup

* Code: branch `feat/expert-stats`; tree `~/vllm-expstats-eeec2b0d38` (SHA
  `eeec2b0d38a40a6443ad0ff30104fdef104cef8e`, `git archive` + `.sha`). The `.so` files
  are those of `~/vllm-hier-2d0f6ccdf9` (csrc, cmake, setup.py and rust are
  identical between `2d0f6ccdf9` and the tree, checked with `git diff`), the
  `_version.py` and `third_party/triton_kernels` come from `~/vllm-hwq-c6b1ff354e`.
  Venv `venv-7.14.0_0.32.0`, per-boot `VLLM_CACHE_ROOT`.
* Driver: `tools/rdna/hetero_moe/t10_run.sh` (lease `ExpertStats.slotB`, HIP -> PCI
  bus check, boot lock and quiet windows through `serve_validate.sh`, KFD eviction
  check per cell, 5 s junction temperature log with the abort rule, AER and load
  gate) and `t10_load_hook.sh` (agentic traffic). Report tool
  `tools/rdna/hetero_moe/t10_report.py` (`itl`, `check`).
* Arms, one fresh boot each, order A1 B1 B2 A2:
  `A` = no `VLLM_EXPERT_STATS_DIR`; `B` = `VLLM_EXPERT_STATS_DIR`, interval 30 s,
  `VLLM_EXPERT_STATS_PAIRS=1` (the collection configuration), torch formulation.
  Cells (identical in every arm, `--ignore-eos --temperature 0`, seed 12345):
  1024/512 at c1, c6, c8 and 16384/1024 at c1, c8. After its cells B1 ran
  4034 s of agentic traffic.
* Agentic traffic: `agentic_bench.py run` behind the Cline system prompt
  (`cline_act.md`), tasks gsm8k, humaneval, mbpp, ifeval, toolcall, needle, 8 workers,
  limit 60, greedy, passes with `--seed 1` to `7` (different item subsets). The
  `decision` task ignores the seed and was left out. 7 passes; the B1 boot in total
  (cells, probes, traffic) computed 8.31 M prompt tokens, generated 534 k tokens in
  207 k MTP verification steps.
* Validity: no KFD eviction jump in any cell or boot (`evicted_ms` +0 on all four
  workers), 0 new amdgpu dmesg faults, PCIe AER line count unchanged (15 before
  and after every arm), junction temperature 58-61 C mean and 82 C maximum over the
  lease (limit 95), no `rdna_ar` wedge.
* A, B arms also ran the standard probes: greedy probe, PPL (4.82-4.89, limit 60) and
  the prefix probe, which prints FAIL in all four arms including the stats-off ones
  (warm speedup 1.8x below its threshold; unrelated to the collector). The
  concurrency probe passed in A1, A2 and B1; B2 reported one FAIL (together, prompt
  0, token 28 of 64, margin 1.375 nats, continuation mean logprob -0.13 against
  -0.13 solo) among 17 near-tie warnings, the classification of batch-shape numerics
  (`[INFERENCE]`: the collector does not touch model tensors); A1/A2/B1 had none.

## 3. Collector validation (items 1-4, 6)

Per arm `t10_report.py check` on `stats/` (49 layers: 48 target + MTP index 0).

| | B1 (4998 s) | B2 (874 s) |
|---|---|---|
| snapshots | 154 | 23 |
| problems (ordering, `sum(counts) vs tokens x top_k`, decode counts vs rows histogram) | none | none |
| target rows, final - snapshot 1 | 8,889,547 | 714,987 |
| server scheduled rows at the last /metrics poll | 8,933,107 | 715,516 |
| difference | -0.49 % (final periodic snapshot 37 s before the poll, load running) | -0.07 % (-529 rows) |
| time-aligned over all snapshots (poll interpolated to the snapshot time) | median +0.16 %, last +0.31 % | median +0.16 %, last +0.04 % |
| MTP layer rows / target rows | 1.028 | 1.073 |

The server's scheduled rows are `vllm:iteration_tokens_total` (prompt tokens
computed + tokens generated) plus the rejected draft tokens, because the verifier
runs `1 + k` rows per request and the metric counts only the accepted ones; the
formula reproduces the collector to 0.07 % where the staleness of the last
periodic snapshot is the only difference. Snapshot 1 holds 2048 rows (the
profile/capture runs, which call the model runner and not `Worker.execute_model`
where the collector ticks) and is subtracted.

Rank agreement: with TP4 + EP4 the MoE layer sees `tp_size = 1`, so the rank-0
filter of the collector passes on all four EP ranks and each writes a file (the
`tp_rank` in every file is 0). The four files agree: same step count at the last
snapshot, per-layer per-expert counts differ by at most 2 rows (B2: 25 of 49
layers differ, relative 2.2e-6 of the layer's decode rows). The difference is a
router near-tie: the hidden states of the ranks are not bitwise identical after the
one-shot all-reduce, so one row of a near-tied logit pair may take the other
expert. `VLLM_EXPERT_STATS_ALL_RANKS=1` (new, debug) records every TP rank with its
`tp_rank` in the meta line; under EP all ranks already record, so the cost of
collection is paid on all four GPUs (they run concurrently, the step time is the
same as with one recording rank).

## 4. Overhead (item 5): ITL, stats off against stats on

Median ITL in ms per cell and arm; decision rule fixed beforehand: the collector has
a resolved cost only if `mean(B) - mean(A)` exceeds both within-condition spreads
(`|A1 - A2|`, `|B1 - B2|`), as in `docs/rdna2/hwq-abba-2026-11-06.md`.

| cell | A1 | B1 | B2 | A2 | mean A | mean B | delta | % | spread A / B |
|---|---|---|---|---|---|---|---|---|---|
| 1024/512 c1 | 22.63 | 28.17 | 28.28 | 22.63 | 22.63 | 28.23 | +5.59 | +24.7 | 0.00 / 0.11 |
| 1024/512 c6 | 42.05 | 47.88 | 48.10 | 41.76 | 41.91 | 47.99 | +6.08 | +14.5 | 0.28 / 0.23 |
| 1024/512 c8 | 44.11 | 50.27 | 50.11 | 44.61 | 44.36 | 50.19 | +5.83 | +13.1 | 0.50 / 0.16 |
| 16384/1024 c1 | 23.13 | 28.78 | 28.67 | 23.25 | 23.19 | 28.73 | +5.53 | +23.9 | 0.12 / 0.10 |
| 16384/1024 c8 | 46.73 | 52.39 | 52.41 | 46.00 | 46.37 | 52.40 | +6.04 | +13.0 | 0.73 / 0.02 |

The stats-off ITL reproduces the hwq ABBA (22.66 ms c1 and 44.49 ms c8 at 4 hardware
queues). The collector adds a constant 5.5-6.1 ms per step independent of batch and
context: one device chain of about 35 small torch kernels per MoE call
(`.to`, compare, `where`, `index_add_`, `ones_like`, the row histogram, the 15
operations of the pair counts) times 49 calls is about 1700 kernel nodes in the
captured graph, about 3.4 us each (`[INFERENCE]`, not profiled). Aggregate output
throughput drops by 16 % at c1, 11-12 % at c6/c8 on 1k prompts and 5 % at 16k c8
(`output_throughput` in `itl.json`; prefill-heavy cells dilute the per-step cost). At that price the collector is a profiling tool, not
something to leave on; section 6 replaces it with 2 launches per call.

## 5. Real statistics, plans, held-out simulation (item 8)

Windows are whole agentic passes of B1's load hook, in collector `elapsed_s`:
`W1` = passes 1-3 (843.2-2694.3 s, 9,099 decode steps, 2,690 prefill steps),
`W2` = passes 4-7 (2694.3-4998.6 s, 11,815 and 3,569 steps). The pairs file is
cumulative and not windowed, so the windows are converted from the JSONL alone.
The cell traffic of B1 (random-token prompts) is outside both windows. Rows: 94 %
of the routed rows are prefill rows (3.3 M per layer in W1 against 0.21 M decode);
decode rows per step 22.8 (about 7.6 sequences x 3 MTP rows).

Skew (decode rows per layer, mean, min and max over the 48 target layers):

| top-N experts of 512 | 8 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|
| W1 mean | 0.099 | 0.251 | 0.388 | 0.585 | 0.831 |
| W1 min / max | 0.044 / 0.174 | 0.148 / 0.347 | 0.260 / 0.513 | 0.447 / 0.730 | 0.732 / 0.938 |
| W2 mean | 0.098 | 0.248 | 0.385 | 0.581 | 0.831 |
| prefill rows, W1 mean | 0.105 | 0.279 | 0.439 | 0.661 | 0.897 |

Plans (`bench_results/2026-11-07_expert-stats-t10/`, exact commands in
`make_plans.sh`; constants of the spec section 7.2 at `30e22d0b1a`):

* H1: one island (W7800 HIP 0 + V620 HIP 1-4, switch c0), `--island-mode replicate`,
  `--hub-active-gib 9.4`, `--hub-vram-gib 44.98`, reserve 12.1, expert budget 26,
  `--max-hub-per-layer 130`, `--hub-fixed-us 30.8`, `--expert-fixed-us 3.1`, MTP
  policy hub. 6,752 hub experts (130 on each of the 48 target layers, all 512 of
  the MTP layer).
* H2: two islands (`0;5` with `1,2,3,4;6,7,8,9`, switches `c0,00`),
  `--island-mode partition`, `--hub-active-gib 5.8`, `--max-hub-per-layer 115`,
  `--hub-fixed-us 20.7`. 11,552 hub experts.

Each plan is solved on one window and simulated (`simulate`, 2000 steps) on both;
the other window is the held-out one. The simulator charges the `hx` broadcast
exchange (every row of the step to every V620 of the hub's group, `2 x (latency + D
x T x hidden / gbps)`; the first version of this table used the per-device row count,
`--exchange rows`, 3-21 % optimistic from 6 rows per step up) and, since the
correlated row model (section 6.1), draws the rows of a step from the model fitted on
the arm-G id trace instead of independently (`--row-model independent` gives the
earlier figures, next to last column). "row" is the plan format's dispatch rule
`(row + layer) % n_cold`, "primary" always uses copy 0:

| plan (solved on) | simulated on | hub hit | step, row (ms) | step, primary (ms) | step, independent rows, row (ms) | spilled / step, row |
|---|---|---|---|---|---|---|
| H1 W1 | W1 (in-sample) | 0.586 | 10.64 | 9.66 | 12.16 | 586 |
| H1 W1 | W2 (held out) | 0.580 | 10.99 | 9.94 | 12.56 | 564 |
| H1 W2 | W1 (held out) | 0.583 | 10.67 | 9.67 | 12.18 | 566 |
| H1 W2 | W2 (in-sample) | 0.582 | 10.97 | 9.94 | 12.54 | 578 |
| H2 W1 | W1 (in-sample) | 0.789 | 6.58 | 6.19 | 7.30 | 1486 |
| H2 W1 | W2 (held out) | 0.784 | 6.76 | 6.35 | 7.51 | 1505 |
| H2 W2 | W1 (held out) | 0.786 | 6.59 | 6.19 | 7.31 | 1459 |
| H2 W2 | W2 (in-sample) | 0.788 | 6.75 | 6.35 | 7.50 | 1532 |

The plans themselves (owner tables, `layers`, plan ids a6fecb0c623d1c78,
107ef238e7cc83d8, d233ec3f425cddd5, 09c4b28fe0ff63ab) are identical to those of the
independent-row version; only `estimate`, `runtime` and `meta` changed: the solver's
objective stays the analytic hit and balance (marginal based), and the row model
enters only the simulated estimates and `runtime.spill_k`. The hub hit changes by at
most 0.006 between in-sample and held-out. The simulated decode hub hit of the plan
on the window it was solved on is 0.586 for H1 W1 against 0.5893 analytic: the
correlated sampler keeps the marginals only approximately (the hottest experts come
out 3-4 % short, section 6.1), a 0.3-point bias; with independent rows it matched to
1e-4. `validate` passes all four plans. The plan's simulated step is 0.895 / 0.897
(H1, W1 / W2) and 0.798 / 0.800 (H2) of the blind `e % n` stripe of the same budget
(hit 0.25); 0.86 and 0.78 with independent rows: correlated rows touch fewer
distinct experts, which helps the stripe more than the plan.

Replicas and dispatch (`estimate.dispatch_variants` of each plan, simulated decode
with correlated rows, H1 W1 / H2 W1): `row` 10.72 / 6.62 ms, `expert` (`(expert +
layer) % n_cold`) 9.79 / 6.24, `primary` 9.73 / 6.23 (with independent rows 12.25 /
7.35, 11.74 / 7.22, 11.66 / 7.21). With `row` dispatch an expert with two cold copies
is streamed on both copies in a step: touches per distinct expert 1.235 (H1) and
1.365 (H2), against 1.098 with independent rows, because the rows that share an
expert (the MTP group, the sequences) are spread over the copies by their row index;
`primary` removes that duplication: decode step x0.907 (H1; 0.906 on W2) and x0.941
(H2; 0.940 on W2), against x0.952 and x0.982 with independent rows. The replicas pay
for prefill instead: the analytic per-device pair-load imbalance (max / mean) with
rows spread over the copies is 1.083 (H1) and 1.057 (H2), against 1.129 and 1.375 with
the primary copy only (unchanged: it does not depend on the decode rows). So the
`row` rule costs 9 % (H1) and 6 % (H2) of the decode step against `primary`, more
than the 5 % and 2 % of the independent-row model, and the replicas earn that back
only in the prefill balance (H2 especially); prefill rows were not traced, so
whether prefill rows share experts like decode rows is open.

`runtime.spill_k` (experts a hub keeps, per hub and layer, at 3 / 6 / 12 / 24 / 48
rows per step; `spill_k_dispatch` = `row`, `spill_k_exchange` = broadcast,
`spill_k_row_model` = `flashnext-mtp2`): H1 W1 9 / 17 / 31 / 53 / 87 (primary 9 / 16
/ 28 / 48 / 79; expert 9 / 16 / 28 / 48 / 80), H1 W2 9 / 17 / 31 / 53 / 88, H2 W1 7 /
11 / 18 / 31 / 52 (primary and expert 7 / 11 / 17 / 29 / 48), H2 W2 the same as W1;
`spill_k_by_dispatch` carries all three rules. The independent-row seeds were 10 / 20
/ 36 / 62 / 102 (H1 W1) and 8 / 13 / 21 / 35 / 58 (H2 W1): the correlated model lowers
K by 10-15 % (both plans, all widths). The model is still 2-10 % above the real ids at
3-24 rows (section 6.1). One sensitivity run (not committed, old `--exchange rows`
cost): H1 without the
130-per-layer cap has 203 hub experts per layer (9,947 in total), hit 0.73 and
10.98 ms (-3.6 % against 11.39), with 2,576 spilled experts per step against 1,169;
it was not repeated with the broadcast exchange. The spill model spreads spilled
experts evenly over the hub's V620s (`[INFERENCE]`, section 9 of the first
document). Both windows come from the same workload mix (the agentic suite behind
one system prompt): the held-out result shows that the solved placement is stable
inside this mix, not that it transfers to another workload; the plans need
recollection when the traffic changes.

## 6. Follow-up T2 (V620 HIP 2-5): fused kernels, final snapshot, id trace

Run 2026-11-10 09:46-12:12 box time on the interim box layout (six V620, no W7800;
HIP 2-5 = buses d9 dc df e2, lease `ExpertStats.b4`, every link Gen4 x8). The
links are narrower than in sections 3-5 (slot B, x16), so the ITL baseline was
re-measured in the same window: arms A1 F1 F2 A2 (ABBA), then C, then G, one fresh
boot each, all on tree `~/vllm-expstats-46868fb5f7` (SHA
`46868fb5f7e41ff3136dd33d4acbe59db212e66d`, md5-checked against `git archive`; `.so`
files of the T2 tree), compiled mode 3, `FULL_AND_PIECEWISE`. Evidence:
`bench_results/2026-11-10_expert-stats-t2/` (`t10.log`, `gpu_test.txt`, `itl.json`,
per-arm summaries, bench JSONs, `check.json`, `trace_check.*.json`; the 52 MB trace
files stay on the box). Validity: no KFD eviction jump in any of the 29 cell
records, 0 new amdgpu dmesg faults in the six boots, junction temperature mean
56-63 C and maximum 82 C over 1,760 samples.

**GPU unit tests** (HIP 2): 27 passed, 0 skipped, 11 s: the fused Triton kernels equal
the torch reference (invalid ids, padding rows, 1 to 300 rows, pairs on and off), and
the counters and the id trace survive CUDA-graph replay in both formulations.

**Overhead, fused kernels (item 5)**, median ITL in ms, A = stats off (A1, A2), F =
fused collector with pairs (F1, F2):

| cell | A1 | F1 | F2 | A2 | mean A | mean F | delta | % | verdict |
|---|---|---|---|---|---|---|---|---|---|
| 1k c1 | 22.54 | 22.96 | 23.02 | 22.46 | 22.50 | 22.99 | +0.49 | +2.2 | resolved |
| 1k c6 | 43.57 | 43.46 | 43.87 | 42.04 | 42.81 | 43.66 | +0.86 | +2.0 | unresolved (<= 2.4 ms) |
| 1k c8 | 45.63 | 45.18 | 46.71 | 45.92 | 45.78 | 45.94 | +0.16 | +0.4 | unresolved (<= 1.7 ms) |
| 16k c1 | 23.06 | 23.49 | 23.39 | 23.09 | 23.07 | 23.44 | +0.37 | +1.6 | resolved |
| 16k c8 | 47.64 | 47.39 | 48.02 | 47.24 | 47.44 | 47.70 | +0.26 | +0.6 | unresolved (<= 0.9 ms) |

The fused collector costs +0.4 to +0.5 ms per step at c1 (+1.6 to +2.2 %, against
+5.5 ms and +25 % for the torch formulation of section 4) and is not resolved
from zero at c6 and c8 (upper bounds 0.9-2.4 ms, 2-5 %). Output throughput -5.0 %
(c1), -2.4 % (c6), -1.7 % (c8), -7.4 % (16k c1), -2.6 % (16k c8) (resolved only for
c6, 16k c1 and 16k c8; the ITL rows are the primary figure).
The stats-off baseline reproduces section 4's (22.5 against 22.6 ms at c1, 45.8
against 44.4 ms at c8): the x8 links cost c8 about 1.3 ms.

**Collector checks on F1, F2, C, G** (`t10_report check`): no problem in any of the 16
rank files; `sum(counts) == top_k x tokens` and decode counts against the row
histogram: 0 bad layers in 49 layers x 4 ranks per arm; the target rows of
`final - snapshot 1` exceed the server's scheduled rows by +0.078 % (F1), +0.076 %
(F2), +0.101 % (C), +0.112 % (G), which is the 9-14 s the last /metrics poll
precedes the final snapshot (the final snapshot is written, so it sees the tail the
periodic ones could not). **Item 7 passes**: every file's last snapshot is newer
than the last periodic one (`final_newer_than_last_periodic`, 16 of 16; elapsed
1,560.9 against 1,558.4 s in F1), with the `close()` fix of `e7fdd4cd02`. Rank
agreement (item 4) in arm C with all four ranks recording: equal step counts (11,597
layer-0 steps in every file), per-expert counts equal or differing by 1-2 rows
(<= 1.2e-6 of a layer's decode rows), the same router near-ties as in section 3.

Probes: PPL 4.78-4.87 in all six arms, greedy probe pass, no fault marker. The
concurrency probe passed in A1, A2, F1, F2 and C; **G had one FAIL** (churn, prompt 1,
token 25 of 64, margin 2.17 nats, continuation mean logprob -0.11 against -0.19 solo)
among 19 near-tie warnings. Together with B2 in section 3 that is two of the six
stats-on arms with one such flip and none of the four stats-off arms; the collector
does not write model tensors, so this is classified as batch-shape numerics
(`[INFERENCE]`) and stays an open risk below. The prefix probe prints its known
FAIL (warm speedup) in all arms.

**Row independence (arm G, follow-up 7)**: fused collector without pairs and the
top-k id ring of layers 0, 12, 24, 36, 47 (8,192 slots; 5,242 decode calls were
recorded, 5,234 with valid rows: 3,206 steps of 24 rows = 8 sequences x 3 MTP rows,
1,113 of 3 rows, the rest 21-22). `trace_check` compares the real ids with the same
number of rows drawn independently from the same layer's marginal (the model of the
simulator and the solver):

| layer | distinct experts per 24-row step, real / independent | shared experts per pair, same MTP group / other groups / independent |
|---|---|---|
| 0 | 159.7 / 179.1 (0.89) | 0.82 / 0.48 / 0.28 |
| 12 | 130.2 / 168.8 (0.77) | 2.41 / 0.77 / 0.38 |
| 24 | 118.2 / 166.0 (0.71) | 3.33 / 0.92 / 0.41 |
| 36 | 103.7 / 157.9 (0.66) | 4.12 / 1.07 / 0.46 |
| 47 | 117.5 / 150.7 (0.78) | 2.87 / 1.10 / 0.63 |

**The independence assumption does not hold**: the rows of one MTP group (a
sequence's verified token and its two drafts) share 3-9 times the experts of two
independent rows in the middle and late layers (1.7-2.9 times at layer 0), and
even rows of different sequences share 1.7-2.3 times; a 24-row step touches 11-34 %
fewer distinct experts than the model predicts (the layers' deep end the most). It
is not an artefact of
mixing the random-token cell with the agentic traffic: the first 1,500 steps (the
cell), the middle and the last 1,700 give the same picture (distinct ratio at layer
24: 0.78, 0.78, 0.75; same-group sharing 3.4, 3.1, 3.5).

The plans evaluated on the real ids against independent rows from the same
marginal (5 traced layers, so absolute step times cover only those layers; H1 W1 and
H2 W1 plans, 3,206 steps of 24 rows; `trace_check.H1.W1.json`, `.H2.W1.json`):

| per step, 24 rows | H1 real | H1 independent | H2 real | H2 independent |
|---|---|---|---|---|
| hub hit | 0.542 | 0.539 | 0.754 | 0.750 |
| hub experts touched (5 layers) | 285 | 378 (x0.76) | 231 | 300 (x0.77) |
| hub experts kept, per hub and layer | 53.0 | 67.9 (x0.78) | 34.4 | 39.5 (x0.87) |
| hub experts spilled (5 layers) | 20 | 38 (x0.53) | 92 | 176 (x0.52) |
| cold distinct experts | 344 | 445 (x0.77) | 194 | 252 (x0.77) |
| cold max touched per V620 | 131 | 148 (x0.88) | 51 | 54 (x0.94) |
| step time (5 layers, us) | 1219 | 1341 (x0.91) | 708 | 765 (x0.93) |

The hub hit is unchanged (the marginal is right), but the number of distinct
experts per step is not: the real step is 7-9 % shorter than the model says, the hub
touches about 24 % fewer experts and the spill decision keeps 13-22 % fewer experts
on the hub. At 3 rows per step (one sequence) the same ratios are 0.75 (H1) and 0.80
(H2) for the kept experts. **Consequence**: the independent-row seeds of
`runtime.spill_k` (10 / 20 / 36 / 62 / 102 for H1, 8 / 13 / 21 / 35 / 58 for H2)
overstate the experts a hub keeps; on the traced layers K(24) is 53 (H1) and 34 (H2)
from the real ids against 68 and 39 from independent rows. The simulator and the
solver now draw correlated rows (next section).

### 6.1 Correlated row model (default of `simulate` and `solve`)

`tools/rdna/hetero_moe/row_model.py` and `sampling.correlated_sample`. A step draws a
base row from the layer marginal; each MTP group (3 consecutive rows, the verified
token and its two drafts of one sequence) keeps each expert of the step's base with
probability `tau` and fills up from a fresh draw; each row keeps each expert of its
group's base with probability `rho` and fills up the same way (`rho = tau = 0` is the
independent model, `--row-model independent`). `rho` and `tau` are fitted per traced
layer by matching the shared experts per row pair, within a group and across groups,
on the 3,206 24-row steps of arm G (rank 0 file, `row_model fit`; the moments match
to 0.01), and interpolated linearly over the layer index (the MTP layer uses layer
47's): layer 0 `rho` 0.24, `tau` 0.62; 12: 0.47, 0.45; 24: 0.56, 0.43; 36: 0.63,
0.42; 47: 0.50, 0.48. The shipped fit is
`tools/rdna/hetero_moe/row_model_flashnext_mtp2.json` (name `flashnext-mtp2`, the
default); `--row-model PATH` takes another fit and every plan records the model name
in `meta.cost_model.row_model`.

Validation against the real ids (`trace_check --row-model`, evidence in
`bench_results/2026-11-10_expert-stats-t2/row_model/`): the fit uses only the pair
moments at 24 rows, so the following are not fitted. Distinct experts per step,
mean over the five traced layers (the widths 3, 6, 12 are the first 1, 2, 4 sequences
of the 24-row steps, 3,206 steps each, so they are not independent captures):

| rows per step | real | independent rows | correlated model |
|---|---|---|---|
| 3 | 23.4 | 28.6 (x1.22) | 23.3 (x1.00) |
| 6 | 42.2 | 53.8 (x1.27) | 43.1 (x1.02) |
| 12 | 74.7 | 97.5 (x1.31) | 76.7 (x1.03) |
| 24 | 125.9 | 164.5 (x1.31) | 129.7 (x1.03) |

Per layer at 24 rows the model is 0.99-1.06 of the real count (layer 36 the worst,
110.3 against 103.7); at 21 and 22 rows (302 and 234 steps) it is 0.97-1.02.

The plans on the real ids against the model, H1 W1 / H2 W1, row dispatch, per hub and
layer unless stated (`kept` = experts the hub computes after the spill rule; the
5-layer step time in microseconds):

| rows | H1 kept real / indep / model | H1 step real / indep / model | H2 kept real / indep / model | H2 step real / indep / model |
|---|---|---|---|---|
| 3 | 11.0 / 14.0 / 11.7 | 334 / 342 / 327 | 8.2 / 9.9 / 8.4 | 246 / 248 / 243 |
| 6 | 18.9 / 24.1 / 20.5 | 487 / 512 / 479 | 13.3 / 16.0 / 14.1 | 322 / 331 / 318 |
| 12 | 31.9 / 40.9 / 35.0 | 758 / 818 / 742 | 21.2 / 24.8 / 22.3 | 463 / 489 / 455 |
| 24 | 53.0 / 67.9 / 57.5 | 1219 / 1341 / 1190 | 34.4 / 39.5 / 35.6 | 708 / 765 / 698 |

With primary dispatch at 24 rows the kept experts are 50.4 / 65.0 / 54.0 (H1) and
32.3 / 38.2 / 33.0 (H2). The model's hub hit is 0.3 points below the real one (the
marginal of the hottest experts comes out 3-4 % short: layer 24's top expert 0.161
against 0.168; a fill ordered by expert weight over-corrected badly and was not
used). Remaining errors, same plans: the model keeps 6-10 % too many experts on the
hub in H1 (real / model 0.94, 0.92, 0.91, 0.92 at 3 / 6 / 12 / 24 rows) and 2-6 % in
H2 (0.98, 0.94, 0.95, 0.97), touches 2-7 % too many hub experts (H1) and spills
too few at 3-12 rows (H1 4.3 real against 1.8 at 3 rows, 5 layers; at 24 rows 20.2
against 18.7 in H1 and 91.8 against 99.6 in H2). Applying these ratios to the model's
`spill_k` gives a calibrated K of about 8-9 / 16 / 28 / 49 (H1 W1, 3 / 6 / 12 / 24
rows) and 7 / 10 / 17 / 30 (H2 W1) (`[INFERENCE]`: the ratios are from five layers,
one workload mix and rows taken from 8-sequence steps; the seeds in the plans are the
uncalibrated model values).

Limits: the fit is at 24-row steps with groups of 3; steps with a partial group (21,
22 rows) use the same parameters; 48 rows per step (16 sequences) is an extrapolation
(the recipe's `max-num-seqs` is 8, so 48 only occurs in a capture size of a larger
recipe); prefill rows are not traced and stay independent in the analytic prefill
balance; rank 0 only (all four ranks record the same counts, section 6); one
workload mix.

## 7. Risks and open decisions

* The torch collector costs 25 % ITL at c1; the fused collector +0.5 ms (+2 %) at
  c1 and 0.2-0.9 ms (unresolved) at c6 and c8. Do not leave either on in production;
  re-collect in a profiling run.
* The row-independence result (section 6): independent rows overstate the distinct
  experts per step by 11-34 % and `runtime.spill_k` by 13-22 % at 24 rows. The
  correlated row model (section 6.1) removes most of it (distinct experts within
  1-6 %, kept experts 2-10 % high, step time within 3 %) but is fitted on 5 layers and
  one workload mix; it replaces the seeds of `spill_k` (10-15 % lower), raises the
  duplicate streaming of the `row` dispatch from 10 % to 24-36 % of the cold touches
  (decode step: primary is 9 % (H1) and 6 % (H2) faster than `row`) and shrinks the
  plans' step-time gain over the stripe (0.86 to 0.895 for H1, 0.78 to 0.80 for H2).
  The hub hit and the plans' relative ranking are not affected (marginal based).
  The step times are MoE-only simulator figures: the engine-level sign of the
  correlated rows depends on how the TP4 anchor is read (HeteroDesign, spec 5.4a:
  +3.4 ms at c8 anchored, -3.0 ms kept), so do not subtract them from a measured
  step. `simulate` also reports `hub_touched_all_per_step` (summed over all hubs;
  `hub_touched_per_step` is the busiest hub's) and `n_eval_layers`, and takes
  `--rows W` for a fixed step width (the sims in `bench_results/` predate these
  fields).
* The held-out hub hit is a within-workload number. A coding-agent-only, a chat or
  a long-context workload may skew differently (`needle` and the agent system prompt
  are in the mix; the sampling was greedy).
* Prefill rows are 94 % of the routed rows; the solver's `--w-decode`/`--w-prefill`
  weights decide how much of the hub budget serves prefill. The plans above use the
  solver defaults.
* Concurrency-probe FAIL in 2 of 6 stats-on arms (B2, G), none in the 4 stats-off arms:
  one churn/together token with a 1.4-2.2 nat margin; `[INFERENCE]` batch-shape
  numerics, not attributed to the collector, not excluded either.
* The final snapshot of B1 and B2 was not written (bug, fixed and validated in
  section 6); their numbers use the last periodic snapshot, at most 30 s stale.
* The reserve and cost constants stay `[INFERENCE]` until the W7800 microbenchmark.
