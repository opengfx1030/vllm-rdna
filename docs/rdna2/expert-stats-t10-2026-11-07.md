# T10: expert statistics on the production server (V620 slot B), real skew and plans

Test T10 of the hetero-MoE spec (`docs/rdna2/hetero-moe-design-2026-11-07.md`,
branch `feat/hetero-moe`): the routing collector of
`docs/rdna2/expert-stats-2026-11-07.md` on the production `flashnext-mtp2` TP4 + EP4
server, GPU validation items 1-8 of that document's section 8, about one hour of
agentic traffic, and the H1 / H2 placement plans solved from the real statistics
and simulated on a held-out window. Box `par1-cs25`, V620 slot B (HIP 6-9, buses
0a/0f/14/17), compiled mode 3, `FULL_AND_PIECEWISE`, no `--monokernel`, MAXBAT 2048,
all under the slot B lease. Sections 1-5 are measured; section 6 is the follow-up
run (T2) with the final-snapshot fix and the fused kernels.

## 1. Result in one table

| Item | Verdict | Evidence |
|---|---|---|
| 1 registration guard never fires | pass | B1 and B2 (collector on, FULL_AND_PIECEWISE captured after the warm-up): no guard `RuntimeError` in `serve.log`, 49 layers (48 target + MTP) registered on every rank |
| 2 counts == tokens x top_k, tokens == server | pass | per layer `sum(counts) == top_k * tokens`; target rows `final - snapshot 1` against the server's scheduled rows: B2 -0.07 % (the last tick precedes the post-load probe), B1 +0.16 % median over 153 snapshots (10 s poll interpolation, section 3) |
| 3 `index_add_` under graph replay | pass (by invariants) | decode counts against the independent row histogram: 0 of 49 layers off over 8.9 M rows (B1) and 0.7 M (B2); no counter ever decreased in 154 snapshots. The literal eager-vs-graph comparison of one prompt was not run (the two invariants plus the server row count are stricter over real traffic) |
| 4 rank agreement | pass | all four EP ranks record (see section 3): equal step counts, per-expert counts differ by at most 2 rows in 10^6-10^7 (<= 2.2e-6) |
| 5 overhead | **fails the budget** | +5.5 to +6.1 ms per step for the torch formulation: +24.7 % ITL at c1, +13 to +14.5 % at c6/c8 (ABBA, resolved far beyond the spread); fused kernels in section 6 |
| 6 snapshot ordering | pass | `seq` +1, `elapsed_s` and every cumulative counter non-decreasing over 154 snapshots (B1) |
| 7 SIGTERM writes a final snapshot | **bug found and fixed** | `close()` raised `RuntimeError: Inplace update to inference tensor outside InferenceMode` at shutdown, so the final snapshot was not written (the last periodic one, up to 30 s old, remains and parses). Fix `close()` runs under `torch.inference_mode()`; a CPU test reproduces the error and passes with the fix; validated on the box in section 6 |
| 8 stats -> solve -> plan -> held-out hub hit | pass | H1 held-out hub hit 0.583 and 0.586 against 0.589 and 0.586 in-sample; H2 0.786 and 0.790 against 0.791 and 0.790 (section 5) |

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
the other window is the held-out one:

| plan (solved on) | simulated on | hub hit | step (ms) | spilled / step | stripe step (ms) |
|---|---|---|---|---|---|
| H1 W1 | W1 (in-sample) | 0.589 | 11.04 | 1177 | 12.84 |
| H1 W1 | W2 (held out) | 0.583 | 11.39 | 1169 | 13.14 |
| H1 W2 | W1 (held out) | 0.586 | 11.05 | 1155 | 12.84 |
| H1 W2 | W2 (in-sample) | 0.586 | 11.37 | 1188 | 13.14 |
| H2 W1 | W1 (in-sample) | 0.791 | 6.16 | 2856 | 8.01 |
| H2 W1 | W2 (held out) | 0.786 | 6.33 | 2916 | 8.17 |
| H2 W2 | W1 (held out) | 0.788 | 6.17 | 2825 | 8.01 |
| H2 W2 | W2 (in-sample) | 0.790 | 6.32 | 2948 | 8.17 |

The hub hit changes by at most 0.006 between in-sample and held-out. The
simulator's decode hub hit of the plan on the window it was solved on equals the
analytic estimate to 1e-4 (H1 W1: 0.5893 simulated, 0.5893 analytic), and the
held-out values stay within 0.6 points of it. `validate` passes all four plans. The
stripe baseline is the blind `e % n` layout of the plan's own budget (hit 0.25).
Step time gains over the stripe: H1 0.86x, H2 0.77x. One sensitivity run (not
committed): H1 without the 130-per-layer cap has 203 hub experts per layer (9,947 in
total), hit 0.73 and 10.98 ms (-3.6 % against 11.39), with 2,576 spilled experts per
step against 1,169: by this cost model the cap costs little time and halves the
spill. The spill model spreads spilled experts evenly over the hub's V620s
(`[INFERENCE]`, section 9 of the first document). Both windows come from the same
workload mix (the agentic suite behind one system prompt): the held-out result shows
that the solved placement is stable inside this mix, not that it transfers to
another workload; the plans need recollection when the traffic changes.

## 6. Follow-up T2: final-snapshot fix and fused kernels

Pending (queued behind the slot B users): GPU unit tests of the fused kernels,
arm C (every rank, torch, 600 s of traffic, SIGTERM final snapshot, rank
agreement on equal final counts), arms F1 and F2 (fused collector, ITL cells, 300 s
of traffic, row check). Results are appended here.

## 7. Risks and open decisions

* The torch collector costs 25 % ITL at c1; do not leave it on in production.
  The fused kernels are untested on the GPU until section 6 completes.
* The held-out hub hit is a within-workload number. A coding-agent-only, a chat or
  a long-context workload may skew differently (`needle` and the agent system prompt
  are in the mix; the sampling was greedy).
* Prefill rows are 94 % of the routed rows; the solver's `--w-decode`/`--w-prefill`
  weights decide how much of the hub budget serves prefill. The plans above use the
  solver defaults.
* The final snapshot of B1 and B2 was not written (bug above); all numbers use the
  last periodic snapshot, at most 30 s stale.
* The reserve and cost constants stay `[INFERENCE]` until the W7800 microbenchmark.
