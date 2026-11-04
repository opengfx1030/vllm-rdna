# Test plan: RidgeFill on V620

**Status: explore, measured; the fill fails quality.** A number goes in a
cell only when a run is recorded in the PR. The compiled TP4 A/B on
a413af6b52 (item 4) is the first boot in which the fill runs: 16K solo
TTFT -7.6 % (rocBLAS map) to -15.3 % (int8 HIP map) against exact, and
long-window PPL +0.72 against exact on every fill arm. The 2026-11-03
compiled "parity" result is withdrawn: that flag-on arm never filled a row
(Measured cells).

Serve through the launcher. Do not hand-write `vllm serve`.

```bash
MODEL=/path/to/checkpoint VENV=/path/to/venv \
  ENABLE_PREFIX_CACHING=0 MTP=0 VLLM_RDNA_RIDGEFILL=0 \
  bash tools/rdna/serve_rdna.sh RECIPE=flashnext
```

The flag-on arm exports `VLLM_RDNA_RIDGEFILL=1` and the paths in
`example_flashnext.env`. Clients and gateways set no RidgeFill variable.
The OpenAI API is unchanged. TTFT is the time to the first streamed chunk.

## Record on every run

| Field | Where |
| --- | --- |
| ROCm version | `rocminfo` or the torch HIP version |
| `mec_fw_version` | the firmware the box reports |
| `GPU_MAX_HW_QUEUES` | the environment of that serve |
| Prompt length | tokens, not a label like "16K" alone |
| Recipe, MTP, prefix caching, `VLLM_RDNA_RIDGEFILL*` | the command line |
| Map sha and checkpoint pins | the refit directory |
| PCIe AER / correctable errors during the run | `dmesg -T` (root) and `lspci -vv` LnkSta of the serving GPUs. A correctable AER storm can collapse the c8 aggregate cells on every arm; treat such c8 numbers as box-contaminated and keep c6 as the clean A/B. |

Empty until a run exists:

| Run | Date | Box | ROCm | mec_fw | GPU_MAX_HW_QUEUES | Prompt tokens | Recipe | Log |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base, exact, compile | 2026-11-03 | par1-cs25 | 7.14.60850 (torch 2.12.0+rocm7.14.0) | not readable without sudo | unset | 4096, 16384 | flashnext, MTP=0, APC off, RIDGEFILL=0 | `~/w4a8_runs/lanes/ridgefill/base` |
| off, flag unset, compile | 2026-11-03 | par1-cs25 | 7.14.60850 | not readable without sudo | unset | 4096, 16384 | flashnext, MTP=0, APC off, RIDGEFILL unset | `~/w4a8_runs/lanes/ridgefill/off` |
| exact, eager | 2026-11-03 | par1-cs25 | 7.14.60850 | not readable without sudo | unset | 4096, 16384 | flashnext, MTP=0, APC off, RIDGEFILL=0, COMPILE_MODE=0 | `~/w4a8_runs/lanes/ridgefill/exact-eager` |
| fill, eager | 2026-11-03 | par1-cs25 | 7.14.60850 | not readable without sudo | unset | 4096, 16384 | flashnext, MTP=0, APC off, RIDGEFILL=1, COMPILE_MODE=0, map `capture` | `~/w4a8_runs/lanes/ridgefill/fill-eager` |
| fill, compile | 2026-11-03 | par1-cs25 | 7.14.60850 | not readable without sudo | unset | none | flashnext, MTP=0, APC off, RIDGEFILL=1, COMPILE_MODE=3 | killed at 150 s (loading weights), no verdict |
| A/B exact, rocblas, f16, i8, exact2, compile | 2026-11-04 | par1-cs25 | 7.14.60850 | not readable without sudo | 2 (recipe) | 4096, 16384; PPL 64 x 16384 | flashnext-mtp2, MTP=0, APC off, tree a413af6b52 | `~/w4a8_runs/lanes/ridgefill/ab_a413af6b52/<arm>` |
| legacy fill diagnostic, compile | 2026-11-04 | par1-cs25 | 7.14.60850 | not readable without sudo | 2 (recipe) | 4096, 16384; PPL 64 x 16384 | flashnext-mtp2, MTP=0, APC off, CACHE_ONLY=0, TAIL=4096, tree a413af6b52 | `~/w4a8_runs/lanes/ridgefill/ab_a413af6b52/legacy` |

## What has to be true

1. **Flag off is the current model.** Same greedy tokens as this branch
   without the draft imported, on a short prompt and on a 16K prompt.
2. **Prompts of 2048 tokens or fewer do not move** when the flag is on
   (tail 2048, nothing is filled).
3. **RULER needles** in the filled region stay close to exact. The paper's
   99.8 against 100 is their MXFP4 run, not a bar this quant must hit.
4. **LongBench v2 or perplexity** on held-out long documents. Report the
   gap. Do not claim the paper's +2.65%.
5. **TTFT** at 9K, 16K, 32K, and 64K, median of at least five paired
   requests. 9K–32K on one server start, alternating with exact, which is
   how the paper measured those three. 64K in the paper is a whole request
   across starts; record both TTFT and request time and do not mix them.
6. **Decode ITL unchanged** on a short generation after the long prefill.
   The map GEMM is prefill-only. If ITL moves, the hook is on the decode
   path and the run stops.
7. **MTP acceptance** is not readable until the dtype pre-check below
   passes. v1 refuses MTP while the flag is on, so this cell is the
   baseline recipe and, later, option C2.

The flag-off path is unchanged in code: the only added branch is a `None`
check on `self._ridgefill`, and the layer's `ridgefill_input` parameter is
`None` on the default path. The GPU greedy-sha comparison was attempted:
base and off differ (PPL 4.81 vs 4.87, probe continuations diverge) because
the recipe's one-shot all-reduce is atomic and not run-to-run
deterministic. A cross-boot sha is not a usable gate here; the CPU tests
cover the flag-off path.

The paper's 1.50–1.60× is an upper reference from two R9700s with experts
in host memory. It is not a V620 target.

## MTP head dtype, before any acceptance rate

Community report, not checked on our checkpoint: the Flash-Next HF MTP head
ships bf16, and on the tcclaviger stack acceptance is silently 0% unless
the head is forced to bf16
(<https://x.com/tunapedia/status/2110968988847243678>).

Before a baseline `flashnext-mtp2` acceptance number, and before any future
fill-plus-MTP2 number, record the dtype of the MTP head tensors in the
quant we actually serve (AWQ, EXL3, or HI2). If that dtype is wrong for
gfx1030, a 0% acceptance rate is a cast bug. Do not attribute it to
RidgeFill. v1 of this draft never reaches this cell with the flag on.

| Quant | MTP tensor | dtype on disk | dtype at runtime | acceptance |
| --- | --- | --- | --- | --- |
| flashnext AWQ, MTP=2, flag off | 31/31 tensors BF16 in `model_mtp.safetensors` (AWQ `ignore` lists `mtp\\..*` and `^mtp.*`) | not run under MTP yet (the loader casts bf16 to fp16 at `--dtype float16`) | not run | not run |
| same, flag on | not run in v1 | | | refused |

## fp16 range (gate, not yet run)

Capture one long prompt. Keep the 10240-wide split state in fp32. Compute
the 2560-wide prediction in fp32. Log max abs of each, per layer 24–47.

| Tensor | max abs | fits fp16 (≤ 65504) | dtype used |
| --- | --- | --- | --- |
| input, 10240, pre-norm | 2.8047 (capture, 32768 rows) | yes | fp32 capture (split state kept fp32) |
| output, 2560 | 36.8438 (max, layer 38, 32768 rows) | yes | fp16 map; fp16 round-trip error 0.0 |

If either side overflows, that tensor stays bf16 and the GEMM for that
apply is bf16. If both fit, fp16 GEMM with fp32 accumulate. The CPU rule
is `math.apply_checked`. This table is the GPU confirmation. The paper
does not say whether the fit saw pre-norm or post-norm inputs, so both
columns stay.

## rocprofv3, map GEMM only

One trace each of fp16 and bf16 at about 8K, 16K, and 32K filled rows
(M), din 10240, dout 2560. Report the GEMM time, not a tok/s invented
from it. bf16 is expected to be slower on gfx1030 (no bf16 dot). If fp16
overflows, the bf16 column is the fallback, not a tuning bake-off.

| M | fp16 GEMM ms | bf16 GEMM ms | overflow? |
| --- | --- | --- | --- |
| ~8192 | | | |
| ~16384 | | | |
| ~32768 | | | |

## Quality and speed

| Arm | RULER filled | LongBench v2 or PPL | TTFT 9K | 16K | 32K | 64K | ITL |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exact, flag off | | | | | | | |
| plain, flag on, refit maps | | | | | | | |
| prompts ≤ 2048, flag on | | | same as exact | | | | |

## Multi-turn, reference only

One trace with a long shared prefix, then a new long document.

| Arm | Prefix-cache setting | TTFT of the new document | Notes |
| --- | --- | --- | --- |
| fill off | off (v1 serve) | | |
| fill off | on | | Reference only. Not the RidgeFill serve. Stock flashnext can already enable prefix caching; this cell is that number, not a RidgeFill win. |
| fill on, v1 | forced off | not run | v1 refuses the boot; `test_env_on_refuses_mtp_and_prefix_cache` covers the refusal |
| fill on, v2 salt | on, salted | not run (`hash_keys.extra_key`, `kv_cache_utils._ridgefill_block_extra_key`) | Only after v1 quality holds. A hit on a filled block must be a fill-on request with the same map sha. An exact request must miss. |

## Negatives

| Case | Expected |
| --- | --- |
| `VLLM_RDNA_RIDGEFILL=1` and MTP=2 | process refuses to boot |
| `VLLM_RDNA_RIDGEFILL=1` and prefix caching, APC mode off | process refuses to boot |
| Released MXFP4 manifest against W4A16 pins | `RidgeFillPinError` |
| Flag unset | hashes and greedy tokens unchanged |

## CPU checks already run

```bash
python3 tests/model_executor/test_ridgefill.py
python3 docs/explore/ridgefill/reference_ridge.py
python3 docs/explore/ridgefill/example_capture_fit.py --show-pin-failure
python3 docs/explore/ridgefill/capture_activations.py --synthetic
```

The real capture (HIP 2-5, TP=4/EP=4, output never under /tmp)::

```bash
MODEL=...capture_activations.py --model $MODEL --tp 4 --prompts 8 --tokens 4096 \
  --output $HOME/w4a8_runs/lanes/ridgefill/capture
```

Capture at the `flashnext` serve shape, TP=4/EP=4. The capture fits one
prompt per request and keeps only the prefill forwards; the decode step has
one row. `--exclude-tail` drops the exact rows from the fit.

The fit recovers a known map, the int8 group-128 dequant matches a hand
product, the range check falls back to bf16 when either the input or the
output exceeds 65504, the flag-off path does not raise on MTP or prefix
caching, the v2 salt is absent on exact blocks, and a vocab mismatch is
refused. No GPU was used.

## Rework items 1-3: row plan, cache-write ledger, compile split

Item 1 builds the row plan **once per prefill forward**, in
`Qwen4ExpForCausalLM.forward` before the compiled model, and every late layer
reuses its two index tensors. Before, each of the 24 late layers rebuilt the
mask: one `positions.cpu()` sync, one `torch.zeros` launch and one fill launch
per filled row, plus two more syncs in the MoE branch (`exact.any()` and the
boolean-index size). After, the only syncs are five reads per forward: two
metadata tensors, the prefill count, the positions authority and the overflow
flag. Decode reads `num_prefills` (a plain int) and syncs zero times.

Item 2 gives filled rows only their cache side effects. The ledger in
`vllm/model_executor/layers/ridgefill/rows.py` assigns every row of each cache
stream (main K/V, indexer raw keys, indexer compressed ring) to the pass that
writes it and proves the assignment partitions the forward: every row written
once, same slot, same absolute position, order preserved per sequence. With
`VLLM_RDNA_RIDGEFILL_CACHE_ONLY=1` the QSA owner runs K/V and indexer K projections
only for filled rows and the layer's attention on the compacted exact rows;
GDN keeps its full-batch conv/state update. Default off until the compiled A/B
runs. The compressed-group cut aligns to `block * compress_ratio` so no group
straddles the two passes.

Counts are analytical from `ridgefill/cost.py`, not measured; each term names
the operation that issues it. 24 late layers, 6 sequences of 4096 or 16384.
The row split reads `positions` once per forward as the authority, because
Flash-Next can carry 3-D M-RoPE positions for vision tokens; the metadata
arithmetic is the fallback and the CPU proof shows the two agree.

| Arm | Host syncs | Host transfers | Row-plan launches |
| --- | --- | --- | --- |
| before (per-layer mask) | 76 | 26 | 295,056 at 4K (2,064,528 at 16K) |
| after (plan once, outside the compiled region) | 5 | 6 | 124 |
| decode (no fill) | 0 | 0 | 0 |
| cache-only extra, 6 QSA layers | 0 | 0 | 36 |
| cache-only extra, all 24 QSA (upper bound) | 0 | 0 | 144 |

The `after` launch count is 4 for the plan (two index uploads, the split-state
gather and its cast) plus 5 per late layer (row replace, map GEMM, MoE select,
MoE scatter, `zeros_like`). The earlier draft counted `zeros_like` once per
forward instead of once per layer; 124 is the corrected number.

CPU proofs, 39 checks, green on the box's venv and on a bare CPU host; the
torch-backed ones skip when torch is absent::

    python3 tests/model_executor/test_ridgefill.py

They cover the row partition and per-sequence order, the metadata arithmetic
matching the positions authority, the subset `query_start_loc`, every cache
stream writing every row once with its slot and position, the compressed-group
straddle and its removal by alignment, and the cached-index replacement being
row-for-row the bool-mask replacement.

Cache-write equivalence is the item-2 proof. A fake paged cache is written once
over all rows and once as filled-then-exact; the two caches are compared byte
for byte. The same check runs with the cut moved mid compressed group, where
alignment is what keeps a group from being written twice. `row_view` is proven
to gather exactly the row-indexed fields (`slot_mapping`, `token_to_req`,
`logical_positions`, `visible_blocks`), leave request-indexed fields
(`block_table`, `seq_lens`, `k_work_metadata`) alone, recompute
`query_start_loc` on device to the same values as the Python authority, and
cache one view per metadata object. The CPU proof stops at the write ledger:
byte identity of the real QSA/GDN kernels is a GPU check, queued with the
compiled A/B, which also records real launch and sync counters with rocprofv3.

Compiled flag-on boot, previous tip: **failed before codegen**. vLLM's
compile runs with `fullgraph`, so the `torch._dynamo.disable`d
`_ridgefill_apply` split point was refused with "Skip calling
`torch.compiler.disable()`d function" on all four workers.

Item 3 removes that split point. The fill decision now happens in
`Qwen4ExpForCausalLM.forward`, outside the compiled region: `begin_forward`
builds the row plan and uploads the two index tensors, and a forward with no
filled rows passes `None`, so its compiled graph is the flag-off graph. The
fill steps are opaque custom ops (`ridgefill_capture`, `ridgefill_predict`,
`ridgefill_replace`, `ridgefill_cache_only`) registered with
`direct_register_custom_op` and fake impls, the PLE pattern. The CPU proof
`test_compiled_fill_branch_is_opaque` compiles the same branch with
`fullgraph=True` and runs it both with and without filled rows. The compiled
flag-on boot is measured below: it succeeds, with 0 graph breaks.

## Map GEMM cost, measured standalone

The map cost decides whether the fill arm can ever beat the exact arm: it is
the only work the fill adds, and the MoE skip is the only work it removes.
Measured with `map_gemm_bench.py` on one V620 (HIP 8), fp16, `a` =
[2560, 10240] per map, 24 maps per forward (7 launches after 3 warm-ups,
median):

| F | 24 x `linear` | T-MAC/s | one stacked `linear` | T-MAC/s |
| --- | --- | --- | --- | --- |
| 2048 | 114.4 ms | 11.26 | 112.5 ms | 11.45 |
| 4096 | 228.2 ms | 11.29 | 225.0 ms | 11.45 |
| 14336 | 791.8 ms | 11.39 | 828.9 ms | 10.88 |

rocBLAS runs this shape at **11.3 T-MAC/s**, exactly half of the 22.8 T-MAC/s
`v_dot2_f32_f16` peak (the dotbench rates are fdot2 22.8, sdot4 46.0, sdot8
91.2 T-MAC/s). At 16K with the 2048-token tail (F = 14336) the map therefore
costs **0.79 s** per request, 10.8 % of the 7.36 s 16K TTFT measured then (an
exact forward, see Measured cells). The served cache-only fill covers less:
its cut rounds down to the compressed-cache unit, KV block 1024 x indexer
compress ratio 4 = 4096 tokens, so a 16384-token prompt fills F = 12288 rows
in six 2048-row chunks (a413af6b52 debug lines: six `filled=2048`, two
`filled=0` per request).

Two measured facts follow. First, one stacked launch is *not* faster than 24
separate ones (828.9 ms against 791.8 ms at 16K), so folding 24 maps into one
GEMM wins nothing on its own. Second, a kernel that reaches 81 % of the
`v_dot2` peak (18.5 T-MAC/s) costs 0.488 s and saves 0.30 s (4.1 % TTFT);
`dot4` at the same efficiency saves 0.55 s (7.5 %) and `dot8` saves 0.67 s
(9.1 %). Those were upper bounds at 81 % efficiency, not measurements;
item 4 below measures the kernels: fp16 12.9-13.6 T-MAC/s, int8
26.8-29.5 T-MAC/s.

## Item 4, the HIP map kernel, measured

Every number here comes from an exact-SHA tree on par1-cs25 (the SHA is
given per table), one V620 (HIP 3) for the kernels, TP4 on HIP 2-5 for the
A/B. Each timed window samples its own process's KFD `evicted_ms`
counters every 0.5 s; a window that overlaps an eviction is invalid and is
rerun (one dotrate window was: 3 evictions while another lane booted).

The first fused kernel (fp16, 128 x 128 tile, 1024 threads, 4 x 4
accumulators) ran at 2.7-2.9 T-MAC/s, 0.24-0.26x rocBLAS. It is replaced.

**Kernel.** `csrc/rocm/ridgefill_map_rdna2.cuh` is one template shared by
the serve ops (`ridgefill_map_rdna2.cu`: `ridgefill_map_rdna2` fp16,
`ridgefill_map_i8_rdna2` int8) and the micro-bench library
(`kbench/mapk.cu`, 18 tile variants), so a bench row is the served kernel.
The 24 maps are one wide NT GEMM: N = 24 x 2560 stacked weight rows,
K = 10240, output `[24, F, 2560]`. The template runs over 4-byte K packs:
`v_dot2_f32_f16` with fp32 accumulation, or `v_dot4_i32_i8` with int32
accumulation and the epilogue `acc * sx[m] * sw[n] + bias[n]`.

The served tile (d4e8668d27, both dots) is 256 threads as 4 x 2 wave32,
8 x 16 accumulators per lane, a 128 x 256 block tile, 64 B K slabs in a
double-buffered LDS (61440 B, rows padded to an odd number of 16 B
chunks), `ds_read2_b64` operand reads, and B packs rotated so pack p of A
and of B sit in different VGPR banks. 230 VGPRs, 0 scratch, 0 spills,
occupancy 4. One slab is 2048 dots, 35 other VALU, 102 SALU/branch,
96 `ds_read2_b64` and 6 `global_load_dwordx4`; 26.7 % of the dots still
read two operands from one VGPR bank after previous-instruction reuse.
All 18 variants compile with 0 scratch and 0 spills (`-Rpass-analysis`).

int8 (`VLLM_RDNA_RIDGEFILL_MAP_MODE=dot4`): `quant_rows_kernel` rotates
each 128-wide K block of the filled rows by the orthonormal Hadamard-128
matrix and quantizes per row (symmetric, absmax / 127), 0.12 ms at
F = 2048. `runtime.quantize_maps_i8` gives the stacked maps the same
rotation and a per-output-row scale at load, in place. dot8 is refused.

**Dot rates** (`kbench/dotrate`, d4e8668d27): `v_dot2c_f32_f16` issues
18.9 T-MAC/s with A, B and D in three VGPR banks and 10.1 with A and B in
one bank; `v_dot4c_i32_i8` 37.7 and 20.2. These are the ceilings; the
22.8 / 46.0 T-MAC/s above came from a latency-bound probe.

**Micro-bench** (`kbench/bench_map.py`, d4e8668d27, 14:47 box time): the
captured maps (`maps.pt`) on captured split states, F filled rows, three
warm-ups and nine timed rounds, median ms (T-MAC/s). rocBLAS per layer is
the shipped path (24 `F.linear`); stacked is one `F.linear` over the
stacked maps. The ratios are against rocBLAS per layer.

| F | rocBLAS per layer | rocBLAS stacked | fp16 kernel | int8 kernel |
| --- | --- | --- | --- | --- |
| 2048 | 121.13 (10.64) | 118.46 (10.88) | 99.61 (12.94), 1.216x | 48.17 (26.75), 2.515x |
| 1024 | 68.63 (9.39) | 59.16 (10.89) | 49.01 (13.14), 1.400x | 22.97 (28.05), 2.988x |
| 512 | 42.38 (7.60) | 28.93 (11.13) | 23.73 (13.57), 1.786x | 10.93 (29.46), 3.876x |

Against the stacked `F.linear` the fp16 kernel is 1.19x / 1.21x / 1.22x
and the int8 kernel 2.46x / 2.58x / 2.65x. The repeat 10 min earlier on
ada368c3b4 (same kernel source) gave fp16 1.211x / 1.399x / 1.767x and
int8 2.505x / 2.987x / 3.878x, and the A/B tree a413af6b52 (16:36) gave
fp16 1.215x / 1.403x / 1.795x and int8 2.512x / 3.004x / 3.896x: the
run-to-run spread is at most 1.6 %. The fp16 kernel reaches 68 % of the
dot2 issue rate, the int8 kernel 71 % of the dot4 rate.

Error against the fp32 product of the same fp16 inputs: fp16 max
8.11e-03, rel RMS 2.07e-04, identical to rocBLAS. int8 max 0.292, rel RMS
2.41e-02 (2.45e-02 at F = 512); the activation quantizer alone is
1.33e-02.

The tile sweep at F = 2048, stacked, x rocBLAS per layer (ada368c3b4):

| Tile | fp16 | int8 |
| --- | --- | --- |
| 8x8, 4x2 waves, b128 (served before d4e8668d27) | 1.051 | 2.113 |
| 8x8, 4x2, b128, no B rotation | 1.043 | 2.093 |
| 8x8, 4x2, b128, register pipelined | 1.011 | 2.029 |
| 8x8, 4x2, b64, pipelined | 1.094 | 2.196 |
| 8x8, 4x2, b64, pipelined, no rotation | 1.086 | not built |
| 8x8, 8-row lanes, 2x4 waves, b128 | 1.049 | 2.113 |
| 8x16, 4x2, b64 (served) | 1.211 | 2.505 |
| 16x8, 8-row lanes, 2x4, b64 | 1.178 | 2.465 |
| 8x8, 2x2 waves, b128 | 1.037 | 2.103 |

**int8 fit error** (CPU on the captured layers, `kbench/int8_schemes.py`
and `kbench/int8_fit.py`). q is the rel RMS of the int8 product against
the fp16 product, fit the rel RMS against the true block input:

| Layer | fp16 fit | W8 per channel | W8A8 per token | W8A8 group 128 | Hadamard-128 W8A8 (served) |
| --- | --- | --- | --- | --- | --- |
| 24 | 0.09965 | q 0.11054, fit 0.14784 | q 0.11475, fit 0.15095 | q 0.01677, fit 0.10102 | q 0.01846, fit 0.10131 |
| 25 | 0.15603 | q 0.11092, fit 0.18999 | q 0.11678, fit 0.19332 | not run | q 0.02142, fit 0.15743 |
| 47 | 0.36268 | not run | q 0.04687, fit 0.36525 | q 0.01683, fit 0.36301 | q 0.02209, fit 0.36326 |

Layers 24 and 25 have outlier input columns (crest factor 53.5 and 46.0),
so plain per-row int8 adds more error than the fit itself has. The
Hadamard-128 rotation spreads each outlier over its block and lands near
group-128 accuracy with a plain per-row epilogue (layer 24: Hadamard-2048
q 0.01354, SmoothQuant alpha 0.5 q 0.06624). Layer 31: q 0.0305, fit
0.4190 against 0.41815 in fp16.

**Bugs fixed**, each caught or pinned by a test:

1. `_predict_fused` handed the HIP op host-resident maps (96af650cc9,
   regression test in `test_ridgefill.py`).
2. `quant_rows_kernel` bit-cast the vector element `v[e]`, which read
   element 0 for every e: each thread's 8 values were quantized as one
   repeated pair (91 % of codes wrong). Copying the word to a scalar first
   fixes it (ada368c3b4). `test_i8_map_matches_its_emulation_and_the_fp32_product`
   caught it at rel 1.1; the int8 GEMM alone matched the exact integer
   product to 2.1e-04 throughout.
3. `bench_map.py` passed the quantizer's row stride in bytes; it takes
   fp16 elements, so F = 2048 read past x and faulted the GPU (ada368c3b4).
4. `test_compiled_fill_branch_is_opaque` fed CPU tensors to ops registered
   for the platform dispatch key only (CUDA on ROCm); it now uses the
   platform device (ada368c3b4).
5. Three stale CPU tests (capture probe worker class, chunk mask
   positions, cached mask) were fixed in 96af650cc9.
6. The fill never ran on a Flash-Next boot: the multimodal entry
   (`Qwen4ExpForConditionalGeneration`) bypassed the row plan, the
   guard-free compiled graph would have served filled forwards as exact,
   and the cut came from the chunk's `seq_lens` instead of the prompt
   length (310e1c2586, e47b35ba7e, a413af6b52; evidence under Measured
   cells).

On the box, the incremental build did not recompile
`ridgefill_map_rdna2.cu` after a header-only change (ninja tracks the
hipified copies). The build script now touches every includer of a
changed header, and each tree records its `_rocm_C` md5 (ada368c3b4
b8f799445ad3, d4e8668d27 ff9cb3c13fa0).

Tests with HIP 3 visible, on d4e8668d27 and again on a413af6b52:
`tests/kernels/core/test_ridgefill_map_rdna2.py` 11 passed (fp16 at rows
1, 100, 129 and 2048 with and without bias; int8 at rows 1, 129 and 2048),
`tests/model_executor/test_ridgefill.py` 44 and 46 passed.

The first kernel exposed two box defects: a clean build is required (a
stale `build/` from another checkout made the extension register
`_rocm_C` twice in one process, and `rm -rf build` fixed it), and the
knobs must be parsed (``bool("0")`` is True; `enabled`/`cache_only`/`debug`
now read through `_env_bool`, so an unset knob is off).

**Compiled TP4 A/B** (a413af6b52, 2026-11-04 16:37-17:47 box time, HIP
2-5, one boot per arm in the order below, `flashnext-mtp2` with MTP off,
prefix caching off, `COMPILE_MODE=3`). Solo TTFT is the median of 16
requests (output length 1) after one 16K warm request; every cell window
was eviction-clean. PPL is `ppl_long.py` on 64 PG-19 windows of 16384
tokens, scoring the last 2047 tokens of each (131008 tokens). With
`VLLM_RDNA_RIDGEFILL_DEBUG=1` every fill arm logged 486 forwards with 2048
of 2048 rows filled per TP rank: 81 requests of 16384 tokens (warm, 16
cell, 64 PPL) x 6 chunks. A 4K request fills nothing: its cut rounds down
to 0.

| Arm | Map path | TTFT 4K | TTFT 16K | 16K vs exact | PPL | PPL vs exact [95 % CI] |
| --- | --- | --- | --- | --- | --- | --- |
| exact | flag off | 1.413 s | 5.741 s | | 7.5441 | |
| rocblas | 24 x `F.linear`, fp16 | 1.418 s | 5.302 s | -7.6 % | 8.2670 | +0.723 [+0.613, +0.835] |
| f16 | HIP dot2, `MAP_FUSED=1` | 1.416 s | 5.208 s | -9.3 % | 8.2599 | +0.716 [+0.604, +0.829] |
| i8 | HIP dot4, `MAP_MODE=dot4` | 1.413 s | 4.863 s | -15.3 % | 8.2678 | +0.724 [+0.612, +0.837] |
| exact2 | flag off | 1.424 s | 5.804 s | +1.1 % | not run | |

Noise floors: exact against exact2 is 0.063 s (1.1 %) at 16K and 0.011 s
at 4K; within a 16K cell the standard deviation is 0.027-0.044 s. The
cross-boot PPL floor is the d4e8668d27 no-fill pair, -0.005
[-0.012, +0.002]. Every fill arm is worse than exact on 64 of 64 windows.

Against the rocBLAS fill, f16 is -0.094 s (-1.8 %) at 16K with PPL -0.007
[-0.015, +0.000], and i8 is -0.439 s (-8.3 %) with PPL +0.001 [-0.007,
+0.009]. The micro-bench predicts -0.129 s and -0.437 s (six 2048-row
chunks per request). The first filled request of a boot also stacks the
maps (warm 16K: rocblas 5.49 s, f16 5.23 s); dot4 quantizes them as well
(6.51 s).

**Diagnostic: the legacy fill on the same rows** (a413af6b52, 19:51-20:07
box time). `VLLM_RDNA_RIDGEFILL_CACHE_ONLY=0` runs the stock QSA attention
on every row (filled rows keep the predicted block input and skip the
MoE), and `VLLM_RDNA_RIDGEFILL_TAIL=4096` cuts each 16384-token window at
12288 without alignment: the same 486 filled forwards per rank as the
cache-only arms. TTFT 1.413 s at 4K and 5.862 s at 16K (filled rows still
run attention). PPL 8.2024: +0.658 [+0.556, +0.762] against exact, worse on
64 of 64 windows, and -0.065 [-0.090, -0.039] against the cache-only
rocBLAS arm, better on 48 of 64.

Two findings follow. Most of the loss is the maps: with the stock attention
path the fill still costs +0.66 PPL. And the cache-only path adds 0.065
that it should not: both arms write the filled rows' caches from the same
predictions, so their exact rows should agree to the noise floor (about
0.01). The cache-only writers (`k_norm` plus `apply_qsa_rope` for the main
K, `_split_qk` plus `_update_and_compress` for the indexer keys) are
therefore not interchangeable with what the stock prefill runs (the fused
QK-norm/RoPE in `_project_qkv_gate`, the HIP indexer epilogue) on this
checkpoint. Which writer differs is not measured: one GPU run that dumps a
filled chunk's main K/V, raw-key and compressed-key cache rows under both
paths would pin it. It does not change the verdict.

**Verdict per path.** Gate: 16K solo TTFT at least 3 % better, PPL within
+0.03.

- **Fill on rocBLAS (path A):** 16K TTFT -7.6 % against exact, PPL +0.723.
  Fails quality by 24x.
- **fp16 HIP kernel (path B, dot2):** 1.22x rocBLAS in the micro-bench, but
  only -1.8 % TTFT against the rocBLAS fill at the same PPL: the map is
  0.73 s of a 5.3 s request. Fails the speed gate.
- **int8 HIP kernel (path B, dot4):** 2.51x rocBLAS, -8.3 % TTFT against
  the rocBLAS fill (-15.3 % against exact), PPL +0.001 against the rocBLAS
  fill. The kernel passes, and the Hadamard-128 int8 costs no measurable
  quality. The fill it accelerates does not: +0.72 PPL against exact.

The fill fails quality at the method level with these maps on this
checkpoint, so nothing ships as a recommendation; the knobs stay opt-in and
off by default. The maps would have to change, not the kernel: the fit
error grows with depth (rel RMS 0.10 at layer 24, 0.36-0.42 at layers 31
and 47) and the fit saw 8 prompts x 4096 tokens. A refit on a much larger
capture, a later split, or a longer exact tail would each have to bring the
long-window PPL within +0.03 before the int8 kernel's -8 % is worth
shipping; none of the three is measured.

## Low-rank maps, measured on CPU

Reduced-rank regression from X's own top singular vectors (64 extra vectors,
ridge 0.03, refit on the first r singular coordinates) on layer 24, CPU,
`~/w4a8_runs/lanes/ridgefill/lowrank/probe2.py`:

| r | rel-RMS vs layer-24 block input |
| --- | --- |
| full map | 0.09997 |
| 256 | 0.58718 |
| 512 | 0.55851 |
| 1024 | 0.52232 |

Truncating the fitted W instead is 0.50 / 0.43 / 0.32 at the same r. Both
are an order of magnitude worse than the full map and barely improve with r,
so a low-rank map is a dead end: the map's rank is not the lever, the
GEMM is.

## Measured cells

Compiled A/B, 2026-11-03 13:42-14:14 box time, TP4 on HIP 2-5, one boot
per arm, `COMPILE_MODE=3 CG_MODE=FULL_AND_PIECEWISE`, prefix caching off,
MTP off, `--temperature 0`. TTFT is the first streamed chunk; prefill
tok/s is the aggregate at 16K. Load 2.6-9.8, PCIe AER count 16 at both
arm starts (no delta), 0 fault markers in either serve log.

| Arm | Solo TTFT 4K | Solo TTFT 16K | Prefill tok/s 16K c6 | Prefill tok/s 16K c8 | ITL c1 | PPL | Probe | Needle filled/tail | Tasks |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exact, flag off, compile | 1.79 s | 7.38 s | 2224.9 | 1505.9 | 19.0 ms | 4.79 | 3/3 | 3/3 + 2/2 | 10/12 |
| plain, flag on, cache-only, compile | 1.80 s | 7.36 s | 2229.0 | 1510.7 | 19.0 ms | 4.85 | 3/3 | 3/3 + 2/2 | 10/12 |
| exact, flag off, eager | 1.79 s | 7.33 s | not run | 975.2 | 66.7 ms | 4.82 | 3/3 | 5/5 | 10/12 |
| plain, flag on, legacy fill, eager | 1.99 s | 8.42 s | not run | not run | 68.8 ms (4/8) | 4.83 | 3/3 | not run | not run |

**Verdict withdrawn (2026-11-04).** The flag-on compiled arm ran the exact
model, for three reasons, each sufficient on its own:

1. The checkpoint resolves to `Qwen4ExpForConditionalGeneration`, whose
   forward called the backbone directly. Only `Qwen4ExpForCausalLM.forward`
   built the row plan, so the session was bound and no forward was ever
   planned: with `VLLM_RDNA_RIDGEFILL_DEBUG=1` on 310e1c2586 the flag-on
   arm logged 0 `[ridgefill] rows=` lines over a 16K request.
2. vLLM compiles `Qwen4ExpModel` once and drops every Dynamo guard
   (`TorchCompileWithNoGuardsWrapper`). The first trace is the 2048-token
   profile run, which is not longer than the 2048-token tail, so it has no
   filled rows, and that flag-off graph would serve every later forward.
3. The row plan cut each request at `seq_lens - tail`, but the QSA
   metadata's `seq_lens` ends at the scheduled chunk. With
   `max_num_batched_tokens` 2048 equal to the tail, every chunk was the
   last 2048 tokens seen so far: on e47b35ba7e (fixes 1 and 2) the flag-on
   arm planned 180 forwards over a 16K request and filled 0 rows.

On d4e8668d27 the flag-on arm's compiled graph calls the exact arm's custom
ops count for count (`moe_forward_shared` 4612, `rdna_hc_mix` 11232) and
no RidgeFill op; its 16K solo TTFT is 5.79 s against 5.78 s and its
long-window PPL delta is -0.005 [-0.012, +0.002]. e47b35ba7e routes both
entries through `Qwen4ExpForCausalLM.run_model`, which sends a forward with
filled rows to the eager forward
(`test_filled_forward_bypasses_the_guard_free_compiled_graph`).
a413af6b52 has `Qwen4ExpModelState` hand the session every request's
prefill length before each forward, and the cut is `prefill_len - tail`
(`test_chunked_prefill_cuts_on_the_runner_prompt_length`). The item 4 A/B
is the first compiled-boot measurement in which the fill runs. The
compiled flag-on row above is therefore an exact measurement. Its PPL pair
4.79 / 4.85 comes from the 261-token probe, which is shorter than the tail
and never fills: it is boot-to-boot noise (flag-off parity below).

The eager flag-on row ran the legacy Python branch, which was slower (+11%
4K, +15% 16K) because of the two defects items 1-2 removed: the per-layer
host waste (76 syncs, 295k launches at 4K) and the full
attention/GDN/o_proj/HC work on rows marked filled. Whether it filled any
row is unrecorded (no debug lines, no tree SHA in `fill-eager/`); with
`max_num_batched_tokens` 2048 and every version of the cut taken from the
forward's own length or the chunk's `seq_lens` (reason 3), it most likely
filled none [INFERENCE]. Its probe PPL 4.83 against 4.82 cannot tell.

The arithmetic. At 16K with the 2048-token tail and the 4096-token cut unit
the fill covers F = 12288 rows (75 %); at 4K the cut rounds down to 0 and
nothing is filled. The map applies 24 maps of 10240 x 2560 per filled row,
so one forward issues F x 6.29e8 MAC: 7.73e12 (15.5 TFLOP) per 16K request,
against the MoE it skips on the same rows. The paper's 1.50-1.60x is two
R9700s with host-resident experts, not this box.

Flag-off parity: the code path is unchanged (`None` checks), but the
recipe's one-shot atomic all-reduce makes greedy text differ run-to-run.
Base vs off: PPL 4.81 vs 4.87 and the probe continuations diverge after
the first tokens. A cross-boot greedy sha is therefore not a usable
parity gate on this box; the flag-off path is structurally identical and
the CPU tests cover it.

`quality_probe.py` is the needle and held-out task probe for one arm:

```bash
python3 docs/explore/ridgefill/quality_probe.py --model flash-next \
  --url http://127.0.0.1:18094/v1/chat/completions
```

It exits 1 when a needle before the tail cut is missed or task accuracy is
below 0.75. Run it on a boot with prefix caching off, so every question
prefills the whole document.

The new checks cover the layer's filled-row replacement, the `Session`'s
fp16 and bf16-overflow map application, the per-sequence filled-row mask
(a chunk that crosses the cut is `mixed`), the torch fit matching
`math.fit_ridge`, the worker extension's hooks and rank file, and the capture script's CPU
synthetic mode. The torch-backed checks run on the box's venv and skip when
torch is absent. The capture script's real W4A16 mode is the GPU follow-up:
it captures layer 24's 10240-wide split state and the 2560-wide block input
to layers 24-47 in fp32, solves the normal equations per layer, and writes
`maps.pt` plus `ridgefill.json` with the per-layer fit error.
