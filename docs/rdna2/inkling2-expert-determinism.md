# `VLLM_RDNA2_MOE_DETERMINISTIC` on the Inkling-2 stock expert path

`vllm/models/inkling2_loop/amd/moe.py` grows one opt-in switch that makes the
routed experts run-to-run bit-identical. It exists because the contribution
contract's token-exact gate *requires* `VLLM_RDNA2_MOE_DETERMINISTIC=1` on both
sides (`.github/CONTRIBUTING.md`, "Token-exact gate"), and until now the flag
only reached the compressed-tensors WNA16 MoE. On Inkling-2 it had no effect,
so a receipt could not judge anything: the base tree did not repeat its own
greedy outputs.

## What the flag changes

- Environment variable `VLLM_RDNA2_MOE_DETERMINISTIC`, read once at import, same
  truth values as the WNA16 path.
- **Unset or `0`** (the shipped default): the dispatch is unchanged, bit for bit.
- **`true` / `1`**: every chunk, decode included, runs the prefill-BM `w13`
  GEMM and a BM `w2` GEMM with `output_topk=0`. Those kernels write one plain
  store per (routed row, column) with a single fp32 accumulator chain, so the K
  slices never reduce through packed-fp16 CAS atomics. The weighted top-k sum
  then runs in fp32 over ascending `k`, element-wise
  (`_sum_topk_rows_f32`), so the result does not depend on how `sorted_ids`
  ordered the routed rows.
- Requires the BM kernels' eligibility (hidden and intermediate multiples of
  32, `VLLM_INKLING2_PREFILL_BM` on). If a model is not eligible the path
  raises instead of silently returning the atomic path's output.

## Cost, measured

Under the flag the expert numerics change (fp32 top-k sum instead of fp16
atomics) and decode uses 32-row BM blocks, so decode is slower. The flag is an
exactness/diagnostic mode, not a production default; `FULL_AND_PIECEWISE`
capture and the production recipe boot unchanged.

## Evidence on the a2 slot (two V620, TP=2)

Same settings on both trees, `VLLM_RDNA2_MOE_DETERMINISTIC=1`, solo cells:

| tree | own-vs-own greedy | unstable cells |
|---|---|---|
| head with this change | **26/26** | none |
| merge-base without it | 16/26 | `solo/0,3,4,5,6` |

With the wired tree on both sides the same receipt is token-exact **PASS,
26/26, rc 0**. That is the property this change claims: the routed experts stop
depending on arrival order. It is not a claim that the rest of the stack (TunableOp
algorithm selection, prefix-cache hit paths) is deterministic.

## Tests

- `tests/models/inkling2_loop/test_moe_deterministic.py` - flag resolution, the
  unset dispatch being the shipped one, the fp32 top-k sum against a reference,
  the scratch buffer reuse, and the ineligible-shape error.
- `tools/rdna/inkling2/moe_det_mutants.py` - mutation battery over those tests.
- Pre-commit and the existing Inkling-2 moe suites.
