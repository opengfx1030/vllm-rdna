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
  ENABLE_PREFIX_CACHING=0 MTP=0 VLLM_RIDGEFILL=0 \
  bash tools/rdna/serve_rdna.sh RECIPE=flashnext
```

The flag-on arm exports `VLLM_RIDGEFILL=1` and the paths in
`example_flashnext.env`. Clients and gateways set no RidgeFill variable.
The OpenAI API is unchanged. TTFT is the time to the first streamed chunk.

## Record on every run

| Field | Where |
| --- | --- |
| ROCm version | `rocminfo` or the torch HIP version |
| `mec_fw_version` | the firmware the box reports |
| `GPU_MAX_HW_QUEUES` | the environment of that serve |
| Prompt length | tokens, not a label like "16K" alone |
| Recipe, MTP, prefix caching, `VLLM_RIDGEFILL*` | the command line |
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
| `VLLM_RIDGEFILL=1` and MTP=2 | process refuses to boot |
| `VLLM_RIDGEFILL=1` and prefix caching, APC mode off | process refuses to boot |
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
`VLLM_RIDGEFILL_CACHE_ONLY=1` the QSA owner runs K/V and indexer K projections
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

CPU proofs, 38 checks, green on the box's venv and on a bare CPU host; the
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
flag-on boot is the next GPU cell; eager-only is not acceptable.

## Measured cells (empty until a run exists)

Eager early signal (COMPILE_MODE=0, not the production recipe). TTFT is
the first streamed chunk; prefill tok/s is the aggregate at 16K, c8.

| Arm | Solo TTFT 4K | Solo TTFT 16K | Prefill tok/s 16K c8 | ITL (decode) | PPL | Greedy agreement | Needle filled/tail | Task accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exact, flag off, compile | 1.79 s | 7.34 s | 1514.3 | 19.0 ms | 4.81 | probe 3/3 | 5/5 (eager) | 10/12 (eager) |
| exact, flag off, eager | 1.79 s | 7.33 s | 975.2 | 66.7 ms | 4.82 | probe 3/3 | 5/5 | 10/12 |
| plain, flag on, refit maps, eager | 1.99 s | 8.42 s | not run | 68.8 ms (partial, 4/8) | 4.83 | probe 3/3 | not run | not run |
| plain, flag on, refit maps, compile | pending: item 3 landed, boot not yet run | | | | | | | |

The eager fill is slower, not faster: 4K TTFT +11%, 16K TTFT +15%,
16K aggregate prefill -13% against exact-eager. The map GEMM is
24 x (10240 x 2560) fp16 per token; the skipped MoE is W4A16 with a
512-wide intermediate, so the map costs more than the MoE it replaces on
this checkpoint. The paper's 1.50-1.60x is two R9700s with host-resident
experts, not this box. The compiled arm is the remaining test; do not
recommend the flag before it lands.

Quality: PPL 4.82 -> 4.83 (+0.2%). The needle/task probe did not run on
the fill arm (the eager bench was stopped for the lane handoff), so the
quality loss is not yet measured beyond PPL and the 3/3 gateway probe.

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
