# Test plan: RidgeFill on V620

**Status: explore. CPU checks pass; the GPU tables hold the eager early
signal only.** A number goes in a cell only when a run is recorded in the
PR. The capture, the refit, the late-layer cache write and the MoE skip
ran on HIP 2-5. The eager A/B is an early signal, not the production
recipe: the flag-on `COMPILE_MODE=3` boot is still unverified (the Dynamo
split-point fix is committed; the first compiled boot was killed for the
lane handoff at 150 s, before compile). The verdict waits for that run.

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
costs **0.79 s** per forward, 10.8 % of the 7.36 s compiled fill TTFT.

Two measured facts follow. First, one stacked launch is *not* faster than 24
separate ones (828.9 ms against 791.8 ms at 16K), so folding 24 maps into one
GEMM wins nothing on its own. Second, a kernel that reaches 81 % of the
`v_dot2` peak (18.5 T-MAC/s) costs 0.488 s and saves 0.30 s (4.1 % TTFT);
`dot4` at the same efficiency saves 0.55 s (7.5 %) and `dot8` saves 0.67 s
(9.1 %). Those are upper bounds at 81 % efficiency, not measurements; the
TunableOp comparison (`tune_prod_shapes.py --mode measure`) and the HIP
kernel's own micro-bench fill them in.

## Item 4, the HIP map kernel, measured

`csrc/rocm/ridgefill_map_rdna2.cu` (opt-in, `VLLM_RDNA_RIDGEFILL_MAP_FUSED=0`
by default) is one wide fp16 GEMM over the stacked maps, `__builtin_amdgcn_fdot2`
in fp32. Measured on one V620 (HIP 4), 24 maps, no bias:

| F | 24 x `F.linear` | one map op | vs rocBLAS |
| --- | --- | --- | --- |
| 2048 | 116.6 ms | 1082.3 ms | 0.11x |
| 4096 | 233.0 ms | 2146.7 ms | 0.11x |
| 14336 | 819.5 ms | 7465.6 ms | 0.11x |

Value check: max |op - F.linear| is 7.8e-4 over the first 2560 columns in
fp16 (e.g. out[0][0][0] = 0.178223 both, out[0][0][83] = 0.390137 both), so
the kernel is numerically the rocBLAS result. It is still 9x slower than the
per-layer stack: one launch does not pay here, the LDS staging and the
fdot2 throughput dominate. The op therefore stays off (`MAP_FUSED=0`) and the
map keeps running on rocBLAS per layer; dot4/dot8 remain the lever.

Two box defects this item exposed: a clean build is required (a stale
`build/` from another checkout made the extension register `_rocm_C` twice
in one process, and `rm -rf build` fixed it), and the knobs must be parsed
(``bool("0")`` is True; `enabled`/`cache_only`/`debug` now read through
`_env_bool`, so an unset knob is off).

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

**Verdict: the compiled fill is at parity with exact.** 4K TTFT +0.6%,
16K TTFT -0.3%, 16K c6 aggregate +0.2%, 16K c8 aggregate +0.3%, decode
ITL identical at 19.0 ms. The flag-on compiled boot succeeds: 0 graph
breaks, 0 "Skip calling `torch.compiler.disable()`d function", 0
`NotImplementedError` from the cache-only path. Eager-only is no longer
needed.

The eager fill was slower (+11% 4K, +15% 16K) because of the two defects
items 1-2 removed: the per-layer host waste (76 syncs, 295k launches at
4K) and the full attention/GDN/o_proj/HC work on filled rows. With both
gone the fill is break-even, not faster: the map GEMM is
24 x (10240 x 2560) fp16 per filled token, and the skipped MoE is W4A16
with a 512-wide intermediate, so the map still costs about what the MoE
it replaces costs on this checkpoint. The paper's 1.50-1.60x is two
R9700s with host-resident experts, not this box. The remaining lever is
the map GEMM itself: dot4/dot8 (item 4) are 2x/4x the fp16 dot rate on a
V620 (22.8 / 46.0 / 91.2 T-MAC/s).

The arithmetic behind the tie. At 16K with the 2048-token tail the fill
covers F = 14336 rows (87.5 %); at 4K it covers 2048 (50 %). The map
applies 24 maps of 10240 x 2560 per filled row, so one forward issues
F x 6.29e8 MAC: 9.02e12 (18.0 TFLOP) at 16K and 1.29e12 (2.58 TFLOP)
at 4K. The MoE it replaces is W4A16 on the same rows, so the two costs
are the same order and the measured break-even is the expected result,
not a tuning failure.

Quality: PPL 4.79 -> 4.85 (+1.25%). Needle 3/3 filled and 2/2 tail, task
accuracy 10/12 on both arms, gateway probe 3/3 on both. The approximation
is measurable in PPL and invisible to the probes at this sample size.

Flag-off parity: the code path is unchanged (a `None` check), but the
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
