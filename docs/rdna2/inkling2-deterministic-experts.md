# Inkling-2 deterministic expert path (`VLLM_RDNA2_MOE_DETERMINISTIC`)

Change class: **opt-in switch**.  With the variable unset (the default) the stock
NVFP4 / MXFP8 expert dispatch is bit-for-bit what ships: the decode kernels with their
fused w2 top-k epilogue, no deterministic scratch, no extra launches.  With
`VLLM_RDNA2_MOE_DETERMINISTIC=1` (or `true`) the routed experts become run-to-run
bit-identical on gfx1030.

## What was wrong

The stock Inkling-2 decode expert GEMMs (`csrc/rocm/moe_nvfp4_gemm_rdna2.cu` /
`moe_mxfp8_gemm_rdna2.cu`) sum the K slices of a row and the top-k rows of a token with
packed-fp16 CAS atomics: the sum rounds in arrival order, and the arrival order changes
between runs (expert blocks execute in a different order depending on scheduling).  On a
48-layer model the difference amplifies: two boots of the same checkpoint, temperature 0,
26 greedy requests, agree on only 16/26 requests, and a single boot does not even repeat
its own requests (3-10 of the 26 differ between its two identical passes).  The WNA16
compressed-tensors MoE already had a deterministic knob (`vllm/envs.py`,
`compressed_tensors_moe_wna16_rdna2.py`), but the stock Inkling-2 experts never read it.

## What the flag does (`vllm/models/inkling2_loop/amd/moe.py`)

* every chunk, decode included, runs the prefill-BM kernels: the w13 kernel stores ONE
  fp32-chained accumulator per (routed row, column) with plain stores (no atomics);
* the w2 stage runs through the BM kernel with `output_topk=0` -- one plain store per
  routed row instead of the atomic top-k epilogue;
* the weighted top-k sum then runs in `_sum_topk_rows_f32`: fp32, ascending k,
  element-wise multiplies and adds (no reduction kernel, no atomics), so the result does
  not depend on how `sorted_ids` ordered the routed rows;
* the `[tokens * top_k, H]` w2 scratch is cached on the workspace (`_det_routed_rows`),
  one allocation per process / graph pool, so the serving forward does not allocate per
  step and the path is HIP-graph capturable (the BM work list is sized by
  `expert_ids.numel()`);
* a layer whose hidden/intermediate sizes the BM kernels cannot run is refused
  (`RuntimeError`) instead of silently running the atomic path, and the offloaded expert
  path (`_forward_hip_offload`) is refused under the flag -- it is not wired.

The same variable, truth values and "resolved once at import" semantics as the WNA16
path: one name for both MoE implementations, never two answers.

## CPU gate

`tests/models/inkling2_loop/test_moe_deterministic.py`:

* truth values (`1`/`true` only), the WNA16 variable name;
* the flag unset resolves to off (subprocess import with the variable removed): the
  default dispatch is the shipped one;
* flag on: 2-token decode forced through the BM w13 + plain-store w2, output equal to the
  fp32 ascending-k sum of the plain stores, and bit-identical across permuted routed rows;
* flag off: the shipped call tuple (decode w13, fused w2 with the atomic epilogue) and no
  deterministic scratch on the workspace;
* refusal of BM-ineligible shapes and of the offloaded path.

`tools/rdna/inkling2/moe_det_mutants.py` kills four mutants (switch forgotten, atomic
top-k kept, decode w13 kept, fp16 top-k sum) with the same test module and restores the
checkout byte-identically.

## GPU evidence (gfx1030, `a2`, 26 greedy requests per boot, flag on)

| run | sides | own-vs-own repeats | verdict |
| --- | --- | --- | --- |
| `det1-...091715Z` | merge-base vs head | head stable `[]`, base `[solo/0,4,5]` | token-exact INCONCLUSIVE |
| `det2-...091715Z` | merge-base vs head | head stable `[]`, base `[solo/3,5,6]` | token-exact INCONCLUSIVE |
| across det1/det2 | same tree twice | **head 26/26 equal**; base 16/26 | head repeats bit-identically |
| `det-hh-...091715Z` | head vs head | both stable `[]` | **token-exact PASS, 26/26** |

The deterministic announce line appears on every head boot and on no merge-base boot.
`pr_receipt.sh` with `EXACT=0` for the switch's own PR: the token-exact gate's premise is
exactly what this change supplies, so the gate is measured as the own-vs-own property
above and the receipt records speed / overlap / hard stall only.

## Cost, stated

* Decode runs the 32-row BM blocks and the separate fp32 top-k sum: slower decode than the
  atomic decode kernels; opt-in, exact-mode only (the default path keeps the shipped
  kernels).  The receipt's speed cells are measured with the flag off, because that is
  what serves.
* The fp32 ascending-k sum is a different (deterministic) rounding than the fp16 atomics:
  outputs change under the flag, by construction.
