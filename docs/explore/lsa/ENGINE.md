# Engine: LSA on this V1 fork

**Status: untested explore draft.** `VLLM_LSA` defaults to off. With the
flag set to `1` the softmax layers still run the attention they run
today. The flag is a server-process environment variable. Clients,
`extra_body`, chat templates, and gateways do not export it and should
not grow a field for it. `docs/explore/lsa/example_qwen38_27b.env` sets
`0`.

Checked against this tree. Flash-Next on ROCm is
`vllm/models/qwen4_exp/amd/model.py`. The 27B dense hybrid is
`vllm/model_executor/models/qwen3_next.py`. DeepSeek-V4 on gfx1030 is
`vllm/models/deepseek_v4/amd/rdna/attention.py`. The stub does not touch
that file.

## What is wired

| Piece | Wired? |
| --- | --- |
| `VLLM_LSA=0` | Yes. Model init records `off`. Forwards do not enter the guard. |
| `VLLM_LSA=1` / `exact` / `guard` | Yes. Init records `exact`. Each softmax forward calls `note_exact_layer` and then the existing attention. Tensors, KV, graphs, and prefix hashes are unchanged. |
| `VLLM_LSA=select` or `tree` | Fail closed. `lsa_mode()` raises `LSAConfigError` at init of Qwen3Next and Qwen4Exp. No half-built selector. |
| Pool, Ward, flat, tree, toy KL | CPU only, `docs/explore/lsa/reference_lsa.py`. Not called from a forward. |
| HIP selector, sparse gather | Not in the tree. Sketch: `sketches/selector_fdot2.cu.txt`. |

DeepSeek-V4 and V4.1 do not call the guard. Setting the variable on
`RECIPE=dsv4-flash` does nothing, which matches Leave in `README.md`.
Selection is not silently enabled there either.

## Where a real indexer would sit

D-LSA, paper Section 3.3: project K and the sum of Q heads into
`d_idx`, L2-normalise, pool, cluster, search, then run the model's own
attention on the index list `J_i`. One list is shared by every query
head. The attention kernel is the existing one, except the paper adds
a kernel for context parallelism (Appendix F.2). This fork's QSA path
already raises `NotImplementedError` for context parallelism
(`vllm/models/qwen4_exp/amd/qsa.py`).

The selector, when it exists, is a new package `vllm/v1/lsa/`
(bucket cache, flat select, the layer list). It is not in this PR.
The flag guard stays in `vllm/model_executor/layers/lsa/` so model
init can fail closed without importing a selector. Official serving
code is the SGLang plugin (`src/lsa/sglang` in their repo, fork tag
`lsa-v0.1.0`). We reimplement against this V1 tree. We do not adopt
that fork as the serve path.

For GLM and Qwen3.8-27B the package would load a file from
`d-lsa-weights-v1` (pins in `README.md`). It would refuse to select
if the file is missing. Flash-Next has no file, so that path stays
exact until a QSA-only distill exists. `VLLM_LSA=1` today does not
read a map and does not change graphs.

Hook, later, not in this PR:

1. After QKV (and after RoPE) in the softmax module:
   `Qwen3NextAttention`, `Qwen4ExpQSAAttention`, or the Glm5Next sparse
   indexer. On the Qwen checkpoints `partial_rotary_factor` is 0.25,
   and the paper reads only the RoPE-free part of Qwen-3.8-27B. Head
   dim 256, so that part is 192 wide. The index dot stays 64 wide.
   On GLM the same hook replaces the kpool indexer. It does not run
   beside it.
2. The selector writes `J_i`. It does not replace the attention op.
3. `RDNA_ATTN` would have to gather paged K and V by that list. It
   does not. Until that gather exists, the only correct behaviour is
   the one this stub has: exact attention.
4. GDN (`linear_attention`) has no softmax KV. `note_exact_layer`
   raises if a caller hands it that type while the guard is armed.
   The forward never does.

QSA already has an indexer (`indexer_n_heads`, budget 2048, compress
4 on the Flash-Next Hub config). A real LSA would replace that
indexer, the way the paper replaces GLM's indexer. Stacking both is
two selectors over one KV. Do not do that.

DeepSeek-V4 already selects (`index_topk` 512, compress ratios 0, 4,
and 128 on `deepseek-ai/DeepSeek-V4-Flash`). V4.1 publishes KV from
`kv_source_layer_ids` `[2, 8, 14, 20]` and candidates from layer 20
(`candidate_source_layer_id`), with more index sources after that.
That is a trained sharing pattern. LSA on top of it is a second index
over a cache the recipe has already compressed. Leave both.

## Tensor parallel

Paper Section 3.3: the bucket cache is replicated. Ranks exchange the
index keys of the new tokens and cluster the same buckets. Query heads
are summed into one list.

On this fork, Qwen KV heads are sharded (or replicated when TP does not
divide the head count). The sum over query heads is an all-reduce of a
64-wide vector, or the index projections are replicated and the local
heads are summed before the all-reduce. Either design waits on a real
selector. This PR does not change `rdna_allreduce` or any GEMM.

## HIP graphs

Recipes serve `FULL_AND_PIECEWISE`. The paper does not capture prefill:
the selection reads list sizes back on the host. Decode is one CUDA
graph per length bucket (`1.125 * 2^e`, `e = 17…23`, plus the max
context). Clustering of a completed block of 512 keys runs before the
replay.

The same split is the one to copy, with HIP graphs in place of CUDA
graphs: no prefill graph, decode graphs kept. Their README requires
`--disable-prefill-cuda-graph` or the server crashes at startup,
because prefill selection runs on the host. Decode stays captured.
Appendix F.1 also captures one decode graph per length range
(`1.125 * 2^e`, `e = 17…23`, plus the longest context). That bucket
split is in the paper. It is not only in the secondary HTML. The
public serve contract to copy first is the README flag: prefill
eager, decode captured. Length buckets are how their decode graphs
avoid launching a 10M grid for a short request.

A host sync inside the captured region breaks this fork's graphs the
same way. The stub has no sync and does not change graph mode.
`VLLM_LSA=1` must not force prefill graphs off, because attention is
still exact. The real selector must.

## Prefix cache and multi-turn clients

The stub does not salt block hashes and does not turn prefix caching
off. A boot with `VLLM_LSA=1` must reuse prefixes exactly as a boot
with the flag unset. `TESTPLAN.md` measures that.

A real selector changes which KV a later token reads, and the bucket
cache has to match the blocks it was built from. Two safe policies,
same choice RidgeFill wrote down:

- Prefix caching off while the selector is on. Simple. Every new turn
  of an agent client re-prefills the whole transcript.
- Salt the block hash while the selector is on, and do not let an
  exact (flag off) request reuse a salted block, or the reverse.
  The salt lives in the server. It is still invisible to the client.
  The client keeps sending the same body. The second turn misses the
  blocks the first turn wrote under the other hash.

Either policy drops multi-turn prefix-cache reuse relative to today's
recipes (`prefix-caching` is on in `flashnext.env` and `27b-awq.env`).
That loss is the point of the trace in `TESTPLAN.md`. Do not "fix" it
by exporting a request field. The gateway has nothing to forward.

## MTP

The stub does not look at MTP and does not refuse it. Refusing would
change who can boot when the flag is on, and the attention would still
be exact. A real indexer has to define what a speculative token does
to the open cluster block and to the window before MTP can stay on.
Until that is written, the implementation should fail closed when
`speculative_config` is set, instead of selecting on draft tokens it
does not understand. That check is not in this PR. The trace for it is
in `TESTPLAN.md`.

## Fail closed

- Unset, `0`, `off`, `false`: exact attention, guard not entered.
- `1`, `exact`, `guard`, `on`, `true`: exact attention, guard entered,
  return value is the layer type unchanged.
- Anything else, including `select` and `tree`: `LSAConfigError` at
  Qwen model init. No fallback that pretends to sparsify.
- No map file is read by this stub. GLM and Qwen3.8-27B maps exist
  and are not loaded. Flash-Next, Qwen3.6-27B, and DeepSeek have no
  map in `d-lsa-weights-v1`.
