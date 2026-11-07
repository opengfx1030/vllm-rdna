# Expert routing statistics and placement solver (hetero MoE)

Branch `feat/expert-stats`. Companion to the hetero-MoE design spec
(`docs/rdna2/hetero-moe-design-2026-11-07.md`, branch `feat/hetero-moe`), which
owns the plan format and the runtime semantics. This document covers the data
side: collecting per-expert routing counts on the V620 TP4 server, and turning
them (plus a byte table and the device budgets) into a placement plan for the
box of 2 PCIe switches x (1 W7800 hub + 4 V620 experts).

Nothing here has run on a GPU. Every claim about GPU behaviour is marked
`[INFERENCE]` or listed under "GPU validation plan".

## 1. What exists

| Piece | Path |
|---|---|
| Collector | `vllm/model_executor/layers/fused_moe/rdna_expert_stats.py` |
| Hook (router output) | `vllm/model_executor/layers/fused_moe/runner/moe_runner.py` |
| Hook (drain, close) | `vllm/v1/worker/gpu_worker.py` |
| Mono K2 stand-down | `vllm/models/qwen4_exp/amd/mono/rdna2/k2.py` |
| Stats IO, solver, validator, simulator, synthetic generator | `tools/rdna/hetero_moe/` |
| Byte table of the real checkpoint | `tools/rdna/hetero_moe/data/flashnext_w4a16_bytes.json` |
| Sample stats and plan (synthetic) | `tools/rdna/hetero_moe/examples/` |
| CPU tests | `tests/model_executor/layers/fused_moe/test_rdna_expert_stats.py`, `tests/tools/rdna/test_hetero_moe.py` |

## 2. The collector

Env vars (all read once at import):

| Variable | Default | Meaning |
|---|---|---|
| `VLLM_EXPERT_STATS_DIR` | unset = off | Output directory; non-empty turns collection on |
| `VLLM_EXPERT_STATS_INTERVAL_S` | 30 | Seconds between snapshots |
| `VLLM_EXPERT_STATS_PAIRS` | 0 | Also count expert pairs (decode only) |
| `VLLM_EXPERT_STATS_PAIRS_EVERY` | 10 | Rewrite the pairs file every N snapshots |
| `VLLM_EXPERT_STATS_DECODE_MAX_TOKENS` | 64 | A call with at most this many rows is "decode", larger is "prefill" |

What is counted, per MoE layer (48 target layers plus the MTP layer, index 48,
kind `mtp`): `counts[phase][e]` = rows whose top-k set contains expert `e`;
`tokens` and `steps` per phase; `rows_hist[n]` = decode calls that had `n` valid
rows; `invalid` = padded rows or out-of-range ids. All counters are cumulative
and never reset, so a window is the difference of two snapshots and there is no
read/zero race with graphs that are replaying.

Where it hooks: `MoERunner._apply_quant_method`, right after `select_experts`
on the modular path, with `topk_ids` as the logical (pre-EP-mapping) expert ids.
That covers the RDNA2 W4A16 path, the gfx1100 port and the stock vLLM quant
methods. Only TP rank 0 of each DP rank records (the router is replicated).
With `--enable-expert-parallel` and no DP, rank 0 sees every token.

How it stays graph-safe: `record()` does fixed-shape device work only (cast ids
to int64, send padding and out-of-range ids to a trash slot, `index_add_` ones
into a preallocated `[phase, E + 2]` tensor). It is `@torch.compiler.disable`
and contains no host synchronisation. Counter tensors are allocated on the
layer's first eager call, which is in vLLM's profile run before graph capture;
if a layer first runs inside a capture the collector raises `RuntimeError`
instead of corrupting the graph. The drain is `tick()`, called once per
`Worker.execute_model` outside the forward pass: every interval it queues
non-blocking copies into pinned buffers plus a `cuda.Event`, and writes the
snapshot on a later tick when the event is complete. No thread issues CUDA calls.
`close()` (worker `shutdown()`) writes a final blocking snapshot.

### Zero cost when off

Proof by inspection plus tests:

1. `COLLECTOR` is `None` when `VLLM_EXPERT_STATS_DIR` is unset or empty.
2. Every use is `if _EXPERT_STATS is not None:`: one call site in
   `moe_runner.py` (per MoE call) and two in `gpu_worker.py` (`tick` per step,
   `close` at shutdown). No tensor, no allocation, no hook registration, no
   thread exists when off. The MoE call is Python-level only: nothing is added
   to a captured graph.
3. `test_rdna_expert_stats.py` checks the first with an AST walk over both
   files (every `_EXPERT_STATS` reference is inside an `is not None` test),
   checks the singleton is `None` when the env is unset, and checks that the
   `record()` source has no `.item`, `.tolist`, `.cpu`, `bincount`, `nonzero`
   or `synchronize`.

### Mono K2 stands down while collecting

`--monokernel` K2 selects the top-k inside a persistent kernel and exposes no
ids (`csrc/rocm/mono_k2_rdna2.cu` exposes `route_ids` only in the offload
variant). When stats are on, the `use` gate in `_k2` is false, so the modules
route and the hook sees ids. The recipe `flashnext-mtp2` does not use
`--monokernel` (only `flashnext-dflash2` does), so for the stats run this
changes nothing. Counting ids inside K2 needs a csrc change and a build; it was
not attempted.

### Do not combine with

EPLB, and the expert cache (`expert_cache_slots`). The older
`VLLM_RDNA_MOE_HIST_DIR` histogram is prefill-only and host-synchronising; it is
unrelated and not reused.

## 3. Formats

### Raw: `expert_stats-dp<D>-pid<P>.jsonl`

Schema `expert-stats/1`: `meta` lines (model, `num_layers`, `num_experts`,
`top_k`, layer list with name, index, kind) and `snapshot` lines (cumulative
`counts`, `tokens`, `steps`, `rows_hist`, `invalid`, `elapsed_s`). The full
field list is the docstring of `rdna_expert_stats.py`. With pairs on, a
`.pairs.npz` sits beside it (`names`, int32 `pairs [L, E, E]`, `a < b`),
rewritten every N snapshots and at close; it is cumulative at its last write
and cannot be windowed.

### Converted: `hetero-moe-expert-stats/1` JSON

`python -m tools.rdna.hetero_moe.stats_io convert RAW... --out stats.json`
subtracts the snapshot at `--t0` from the one at `--t1` (`elapsed_s`), merges
several files with `PATH:WEIGHT`, and writes: `model`,
`window{t0,t1,engine_commit,workload}`, `phases`, `layers[{layer,decode{counts},
prefill{counts}}]`, `step_hist.tokens_per_step` and, when a pairs file is
present, `coact{file, dtype uint32, shape, phase decode}` pointing at a
`.coact.npz` next to the JSON. The MTP layer gets layer index
`num_target_layers + idx`. `stats_io show` prints a skew summary (share of
rows covered by the top 8/16/32/64/128 experts per layer). Truncated final
lines (a killed server) are skipped.

### Plan: `hetero-moe-placement/1`

Defined by the design spec; produced here. Fields: `format`, `version`,
`model`, `plan_id`, `stats_sha256[]`, `expert_bytes`, `devices[{id, role,
arch, hip, switch, expert_budget_bytes}]`, `islands`, `cold_complete`,
`layers[{layer, owners[E][...]}]`, plus `source`, `notes` and an `estimate`
block. `island_mode` is `replicate` (default): `owners` is written for island 0
and island k maps by position; every island is a full copy (one engine per
switch). In `partition` mode `owners` holds global device indices, every expert
lives in exactly one island and each island's V620s hold that island's experts
(`cold_complete` per island): TP2 across the hubs, EP2. Per expert:
hub device indices first, then expert-role indices with the primary first and
replicas ascending. Runtime dispatch: a hub copy wins; otherwise the cold copy
is `(row + layer) % n_cold_copies`. With `cold_complete` (default) every expert
keeps at least one V620 copy and hub copies are an extra cache.

`python -m tools.rdna.hetero_moe.validate plan.json [--stats stats.json]`
checks: format and version; device ids and HIP indices unique, roles `hub` or
`expert`; islands partition the devices, share one role order and stay on one
PCIe switch; one owner list per expert and no duplicate layers; owners inside
island 0 (`replicate`) or exactly one island (`partition`), no duplicates, hub
indices before expert indices, replicas ascending;
one expert-device copy per expert under `cold_complete`; primaries balanced
within one per device and layer; hub and expert devices within their byte
budgets (hub sharing mode taken into account); every stats layer present when
`--stats` is given. Exit 0/1.

## 4. The solver

`python -m tools.rdna.hetero_moe.solve --stats stats.json --out plan.json`

Inputs: stats (JSON or raw JSONL, optional `--t0/--t1`), the byte table (default
the committed one: `expert_bytes` 2,534,400 B packed int4 plus scales, 48 layers,
512 experts, hub active part 10,076,534,296 B, MTP experts bf16 9,830,400 B
each), and the topology (default: hubs HIP 0 and 5, experts HIP 1-4 and 6-9,
switches `c0`, `00`).

Budgets. Hub expert budget = hub VRAM (`--hub-vram-gib`, 44.98: the W7800 48 GB)
minus the active part (`--hub-active-gib`, default the byte table's whole-model
value, about 5.8 for a TP2 hub) minus a reserve (`--hub-reserve-gib`, 12.1: 4.1
non-torch and workspace plus 8 GiB of KV) or `--hub-expert-budget-gib` directly.
V620 expert budget 26 GiB (`--expert-budget-gib`: 29.98 minus non-torch,
workspace and graphs, the optional lm_head shard and a margin). The hub budget
is balance-limited rather than capacity-limited, so `--max-hub-per-layer`
defaults to 130 (115 per island with `--island-mode partition`). With the
defaults a hub's budget is 23.5 GiB and a V620 holds its 128 primaries per
layer (14.5 GiB) plus 11.5 GiB of replicas.

Algorithm:

1. Step shape: distribution of rows per decode step from `step_hist` (a step
   with `r` rows touches an expert iff any of its rows selected it).
2. Island mode `partition` only: each layer's experts are split over the islands
   by LPT on expected touches with equal counts (256 and 256), and steps 3 to 6
   run per island on its own experts, with its own hub and V620 budgets.
3. Hub set: global greedy by marginal expected benefit per byte (decode weight
   1.0, prefill 0.05, adjustable), clamped to `[--min-hub-per-layer,
   --max-hub-per-layer]`, so hot layers get more slots than flat ones. MTP
   policy (`--mtp-policy`, default `hub`): `hub` (all MTP experts on the hub,
   taken from the budget first; each hub holds its island's half under
   `partition`), `omit` (not placed), `solve` (treated as a layer). MTP expert
   size: `--mtp-expert-bytes`, default a target expert's 2,534,400 B (W4A16
   requantised, `[INFERENCE]` same geometry); 9,830,400 for the bf16 checkpoint.
4. Hub mode for an island with several hubs: `replicate` (default, same set on
   each), `shard` (each holds a slice of every hot expert), `partition`
   (disjoint sets, sizes within one and balanced over layers).
5. V620 primaries: longest-processing-time assignment on expected distinct
   touches under the step shape, exactly E/D experts per device per layer.
   Optional co-activation refinement (pairs file): swap search on
   `max_d(mu_d + z*sigma_d)`.
6. Replicas: hottest cold experts first, round by round, onto the least-loaded
   V620 without a copy, until the byte budgets are spent.

Determinism: no randomness in the solve; the plan id is a hash of the content.
The same input gives the same bytes (tested, and the sample below reproduced
the same id twice).

No stats: `--stats` omitted gives a uniform fallback (hub holds `e % 4 == 0` or
an evenly spaced stripe when the budget is smaller, owner `(e + e // D) % D`, no
replicas, `source: uniform-fallback`). It is also the "stripe" baseline in the
estimate block.

### Simulator

`python -m tools.rdna.hetero_moe.simulate plan.json --stats stats.json`
draws decode steps from the measured shape and marginals (systematic sampling
that matches each expert's inclusion probability, so no co-activation) and
reports hub hits, the cold devices' touched experts per layer and the max over
devices (the per-layer critical path). The step time is
`sum_l max(hub_fixed + hub_touched * hub_us, comm + expert_fixed + max_touched *
expert_us)` with comm = `2 * (latency + rows * 5120 B / gbps)`. Defaults are
HeteroDesign's constants, all `[INFERENCE]` until the W7800 microbenchmark:
`--hub-us 3.7`, `--expert-us 6.7`, `--hub-fixed-us 0`, `--expert-fixed-us 0`,
`--latency-us 13.3` (per direction), `--gbps 25.0`. Per-row routing is the
optimistic variant: the runtime sends the same rows to all four servers.

Spill: a touched hub-resident expert may run on its V620 copy instead
(`cold_complete`). The simulator lets each hub spill the number of experts that
minimises the layer time, spreading them evenly over its V620s, and reports
`step_us` (with spill), `step_us_no_spill` and `hub_spilled_per_step`. Plans that
are not
`cold_complete` are never spilled. The analytic
`cold_touched_critical_per_step` is a max of expectations, a lower bound on the
true per-step max; use the simulated `cold_max_touched_per_step` (the solver's
`gain_vs_stripe.simulated_cold_max_touched_ratio` and
`simulated_step_us_ratio`) when comparing.

## 5. Sample plan (synthetic)

`tools/rdna/hetero_moe/examples/sample_stats_synthetic.json` and
`sample_plan_synthetic.json` come from:

```bash
python -m tools.rdna.hetero_moe.synth --out /tmp/raw --kind zipf --alpha 1.1 \
    --layers 48 --experts 512 --top-k 10 --decode-rows 20000 \
    --prefill-rows 50000 --clusters 16 --beta 2.0 --seed 7
python -m tools.rdna.hetero_moe.stats_io convert /tmp/raw/expert_stats-dp0-pid1.jsonl \
    --out tools/rdna/hetero_moe/examples/sample_stats_synthetic.json \
    --workload synthetic-zipf1.1 --engine-commit synthetic
python -m tools.rdna.hetero_moe.solve \
    --stats tools/rdna/hetero_moe/examples/sample_stats_synthetic.json \
    --out tools/rdna/hetero_moe/examples/sample_plan_synthetic.json
python -m tools.rdna.hetero_moe.validate \
    tools/rdna/hetero_moe/examples/sample_plan_synthetic.json \
    --stats tools/rdna/hetero_moe/examples/sample_stats_synthetic.json
python -m tools.rdna.hetero_moe.solve --island-mode partition --hub-active-gib 5.8 \
    --stats tools/rdna/hetero_moe/examples/sample_stats_synthetic.json \
    --out tools/rdna/hetero_moe/examples/sample_plan_synthetic_partition.json
```

The data is synthetic (a Zipf alpha of 1.1 per layer): it demonstrates the
tooling and is not evidence about the real model. A second plan,
`sample_plan_synthetic_partition.json`, is the same solve with
`--island-mode partition --hub-active-gib 5.8`. Results (1,000 simulated steps,
mean 3.1 rows per decode step, HeteroDesign's cost constants):

| Plan | Hub experts | Hub hit (decode) | Cold max touched / step | Step us (spill) | Step us (no spill) |
|---|---|---|---|---|---|
| `replicate` (`e08c31dcd2404d90`) | 6,240 per hub | 0.80 | 137 | 2,492 | 3,384 |
| stripe baseline | 6,144 | 0.25 | 329 | 3,544 | 3,544 |
| `partition` (`85307050c2418c2d`) | 11,040 over 2 hubs | 0.88 | 73 | 1,875 | 2,286 |
| stripe baseline (partition) | 3,072 per hub | 0.23 | 219 | 2,802 | 2,802 |

Both plans are hub-bound without spill (a hub touches about 18 distinct experts
per layer), which is why the spill rule matters: with it the simulated step is
0.70x (replicate) and 0.67x (partition) of the stripe baseline. The real gain
depends entirely on the real skew.

## 6. Skew evidence

No routing trace of the Flash-Next model exists in this repository or on the
`par1-cs25` box (searched read-only), so the real skew is unmeasured, and this
work exists to measure it. The only indirect datum is
`docs/rdna2/moe-offloading-target.md`: an offline real-row decode sweep with 96
cache slots of 256 local experts per rank and layer reached 88.24% expert-demand
hits (100 calls per cache, starting empty). That is a short-window cache
measurement of a different model split, not a cumulative frequency curve, so it
only shows that decode routing has locality. `[INFERENCE]`: the published MoE
literature generally reports skewed and workload-dependent expert popularity;
none of it is Flash-Next and no figure from it is used here. A 512-expert,
top-10 model with a shared expert may well be flatter than a Zipf alpha of 1.1;
the uniform case is covered by the tests (hub gain approaches the stripe
baseline, as it should).

## 7. GPU collection command (V620 TP4, about 1 h of agentic traffic)

Run on the V620 island. The recipe is `flashnext-mtp2` (TP4, expert-parallel,
MTP 2, FULL_AND_PIECEWISE graphs, `--max-num-seqs 8`, port 18096). Confirm the
V620 HIP ids with `rocm-smi` first; the ABBA runs used 6-9.

```bash
export VLLM_EXPERT_STATS_DIR=/data/expert-stats/agentic-1h
export VLLM_EXPERT_STATS_INTERVAL_S=30
export VLLM_EXPERT_STATS_PAIRS=1
mkdir -p "$VLLM_EXPERT_STATS_DIR"
HIP_VISIBLE_DEVICES=6,7,8,9 VENV=<venv> \
    bash tools/rdna/serve_rdna.sh RECIPE=flashnext-mtp2 MODEL=<flash-next checkpoint>
```

`PRINT=1` shows the assembled environment (it lists the `VLLM_EXPERT_STATS_*`
variables) and command without launching. Then drive traffic from a client
(the hi2 checkout and the agent system prompt are those of
`docs/rdna2/w4a8-quality-2026-11-04.md`):

```bash
timeout 3600 bash -c 'i=0; while :; do i=$((i+1));
  python tools/rdna/port_v031/agentic_bench.py --hi2 <hi2 checkout> \
    --system <cline_act.md> run --url http://127.0.0.1:18096 \
    --model flash-next --tasks gsm8k,toolcall --limit 120 --workers 8 \
    --out /data/expert-stats/agentic-1h/client-$i.json; done'
```

Stop the server with SIGTERM (so `close()` writes a final snapshot; after a kill
the last interval snapshot, up to 30 s old, is what remains). Then:

```bash
D=/data/expert-stats/agentic-1h
python -m tools.rdna.hetero_moe.stats_io convert $D/expert_stats-dp0-pid*.jsonl \
    --t0 120 --out $D/stats.json --workload agentic_bench --engine-commit <sha>
python -m tools.rdna.hetero_moe.stats_io show $D/stats.json
python -m tools.rdna.hetero_moe.solve --stats $D/stats.json --out $D/plan.json
python -m tools.rdna.hetero_moe.validate $D/plan.json --stats $D/stats.json
python -m tools.rdna.hetero_moe.simulate $D/plan.json --stats $D/stats.json
```

`--t0 120` drops warm-up. To check the skew is stable, convert two disjoint
windows (`--t0/--t1`), solve on one and `simulate` the plan on the other.

## 8. GPU validation plan

1. Boot with the env above under FULL_AND_PIECEWISE: no `RuntimeError` from the
   registration guard (it fires if a layer first runs during capture; the
   profile run is expected to precede capture, `[INFERENCE]`).
2. Per layer and phase, `sum(counts) == tokens * top_k` and `tokens` equals the
   valid rows the server processed (compare with the server token counters;
   MTP-rejected draft rows are counted).
3. `index_add_` with int64 indices into a counter tensor on gfx1030 under graph
   replay: counts monotone, no lost updates (single-request deterministic
   prompt, compare with the eager run of the same prompt).
4. Rank 0 versus another rank agree (temporary debug run, both ranks recording).
5. Overhead: ITL and aggregate throughput, stats on versus off, `--monokernel`
   off in both arms, one fresh boot per arm. `[INFERENCE]` 49 small
   `index_add_` launches per step.
6. Snapshot ordering under Model Runner V2: the pinned copies are issued on the
   current stream; if compute runs on another stream a snapshot may lag by one
   step (harmless for the cumulative counters).
7. SIGTERM writes a final snapshot; a kill leaves a parseable file.
8. Converted stats feed `solve` and the plan validates; the simulator's hub hit
   on a held-out window matches the real stats of that window.

## 9. Risks and open decisions

Risks:

- With `--monokernel` the K2 path is off while collecting, so timing differs.
  Not an issue for `flashnext-mtp2`.
- EPLB and the expert cache must be off.
- The pairs counters take 1 MB of device memory per layer on rank 0 (about
  49 MB over 49 layers, `[INFERENCE]`) and the pairs file is not windowable.
  The simulator samples from marginals and ignores co-activation. The
  co-activation refinement gain measured on clustered synthetics is modest
  (about 4% cold max touched, in-sample).
- The hub reserve (12.1 GiB: 4.1 non-torch and workspace plus 8 GiB of KV) and
  the cost constants are HeteroDesign's `[INFERENCE]` values until the W7800
  microbenchmark.
- The analytic estimate is optimistic at small touched counts; the simulator is
  the reference.
- The simulator's spill model is a water-filling approximation (spilled experts
  spread evenly over a hub's V620s); the real runtime spills to the owner devices.
- The synthetic sample says nothing about the real skew.

Open decisions for the user:

- Hub KV and workspace reserve.
- MTP experts: `hub` policy with W4A16 requantised experts (HeteroDesign's
  decision; gate: acceptance length at least 97% of bf16, else bf16 on the hub,
  `--mtp-expert-bytes 9830400`, 4.69 GiB per hub, or half of that per hub under
  `--island-mode partition`).
- Island mode: `replicate` (one engine per switch) or `partition` (TP2 across the
  hubs).
- Whether to extend Mono K2 to count ids (csrc work).
