# Test plan: RidgeFill on V620

**Status: explore. CPU checks pass; the GPU tables are empty.** A number
goes in a cell only when a run is recorded in the PR. The CPU checks at the
bottom have been run; they do not use a GPU. The activation-capture script
and the late-layer wiring are in the PR. The HIP 2-5 capture and boot cells
are the next step.

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

Empty until a run exists:

| Run | Date | Box | ROCm | mec_fw | GPU_MAX_HW_QUEUES | Prompt tokens | Recipe | Log |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| | | | | | | | | |

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
`None` on the default path. The GPU greedy-sha comparison is still empty.

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
| flashnext AWQ, MTP=2, flag off | | | | |
| same, flag on | not run in v1 | | | refused |

## fp16 range (gate, not yet run)

Capture one long prompt. Keep the 10240-wide split state in fp32. Compute
the 2560-wide prediction in fp32. Log max abs of each, per layer 24–47.

| Tensor | max abs | fits fp16 (≤ 65504) | dtype used |
| --- | --- | --- | --- |
| input, 10240, pre-norm | | | |
| output, 2560 | | | |

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
| fill on, v1 | forced off | | |
| fill on, v2 salt | on, salted | | Only after v1 quality holds. A hit on a filled block must be a fill-on request with the same map sha. An exact request must miss. |

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

The fit recovers a known map, the int8 group-128 dequant matches a hand
product, the range check falls back to bf16 when either the input or the
output exceeds 65504, the flag-off path does not raise on MTP or prefix
caching, the v2 salt is absent on exact blocks, and a vocab mismatch is
refused. No GPU was used.

The new checks cover the layer's filled-row replacement, the `Session`'s
fp16 map application and filled-row mask, and the capture script's CPU
synthetic mode. The torch-backed checks run on the box's venv and skip when
torch is absent. The capture script's real W4A16 mode is the GPU follow-up:
it captures layer 24's 10240-wide split state and the 2560-wide block input
to layers 24-47 in fp32, solves the normal equations per layer, and writes
`maps.pt` plus `ridgefill.json` with the per-layer fit error.
