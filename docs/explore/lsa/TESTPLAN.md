# Test plan

**Status: untested explore draft.** CPU cells below were run on this
branch. Every GPU cell is empty on purpose. Do not fill them from the
paper's H200 or B300 tables.

`VLLM_LSA` is set in the server environment before the launcher. The
client body is identical in every row. No request field, no gateway
header, no `extra_body` key.

Streaming TTFT is the time from the request leaving the client to the
first SSE chunk. ITL is the gap between later chunks. Report prefill
tok/s, decode tok/s per request, TTFT (s), and ITL (ms), at c=1 and
c=8, for the lengths that fit.

## CPU, this branch

```bash
python3 tests/model_executor/test_lsa.py
python3 docs/explore/lsa/reference_lsa.py
python3 docs/explore/lsa/example_distill.py
```

| Check | Result |
| --- | --- |
| Pool sums, Ward mutual merge, `n_b_max` skip | pass |
| Flat select prefers the matching direction | pass |
| Tree beam stays in the matching half | pass |
| Hub layer rhythm: 12 softmax layers in 48, 16 in 64 | pass |
| KV bytes and the paper's 160 MiB centroid product | pass |
| Toy KL falls with frozen Q,K | pass |
| `VLLM_LSA` unset is `off`; `1` is `exact` and returns the layer type | pass |
| `VLLM_LSA=select` raises | pass |
| Armed guard rejects `linear_attention` | pass |

No GPU, no model load, no eval.

## Accuracy, when a map exists

Greedy. Flag off is the baseline. Order is the released maps first.
Do not run a distill to produce these rows.

| Step | Model | Map | Lengths | What to record |
| --- | --- | --- | --- | --- |
| 1 | GLM-5.3-Flash, when a boot exists | `glm53_flash__long-16k-1M-heavy-d2048-lr3e-4-latent.pt` | 128k, 262144 | RULER, needles, LongBench v2, and whether W4A16 / EXL3 / HI2 Q/K pick the same keys as bf16. Empty until someone can boot GLM on gfx1030. No recipe in `tools/rdna/recipes/`. |
| 2 | Qwen3.8-27B, `27b-awq` then `27b-exl3` | `qwen38_27b__long-16k-1M-heavy-key-and-query-d2048.pt` | 128k, 262144 | Same cells. This is the first model that already has a recipe. |
| 3 | Flash-Next QSA layers only | none yet | 128k, 262144 | Only after a QSA-only distill. GDN layers stay on the baseline. |

| Model | 128k RULER | 262144 RULER | Needle | LongBench v2 | Quant vs bf16 keys |
| --- | --- | --- | --- | --- | --- |
| GLM, loaded map | | | | | |
| Qwen3.8-27B bf16, loaded map | | | | | |
| Qwen3.8-27B AWQ, loaded map | | | | | |
| Qwen3.8-27B EXL3, loaded map | | | | | |
| Qwen3.8-27B HI2, if that checkpoint is the one served | | | | | |
| Flash-Next, after distill | | | | | |

Do not schedule 1M or 10M. PaperTrail (CC BY 4.0, Sukrit Sumant) is
optional and is not the first V620 pass. The paper's own subsets
(RULER 100 prompts to 1M, 20 after; LongBench v2 60 questions) are
what they reported. Our first pass is the two lengths above.

## Speed, streaming TTFT against prompt length

One request, then c=8. Stream the completion. TTFT is the first SSE
chunk. Same prompt text at each length, padded or repeated so the
token count is the row. Client body has no LSA field.

| Model | Prompt tokens | Output | c | Flag | TTFT (s) | ITL (ms) | Prefill tok/s | Decode tok/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3.8-27B | 1024 | 512 | 1 | off | | | | |
| Qwen3.8-27B | 1024 | 512 | 1 | 1 (exact stub) | | | | |
| Qwen3.8-27B | 16384 | 1024 | 1 | off | | | | |
| Qwen3.8-27B | 16384 | 1024 | 1 | 1 | | | | |
| Qwen3.8-27B | 32768 | 1024 | 1 | off | | | | |
| Qwen3.8-27B | 131072 | 1024 | 1 | off | | | | |
| Qwen3.8-27B | longest that fits | 256 | 1 | off | | | | |
| Qwen3.8-27B | 1024 | 512 | 8 | off | | | | |
| Qwen3.8-27B | 16384 | 1024 | 8 | off | | | | |
| Flash-Next | 16384 | 1024 | 1 and 8 | off | | | | |
| Flash-Next | 262144, only if the pool accepts it | 256 | 1 | off | | | | |

The `VLLM_LSA=1` rows must match the off rows. The stub does not change
the forward. A real selector gets the same grid later, and the rows
that decide Take versus Later are 131072 and 262144. Leave those cells
blank until the gather exists. Do not copy 72.4× or 9.2× into this
table. Those are the paper's H200 numbers at 10M.

## Multi-turn prefix-cache trace

Same failure mode as RidgeFill. Agent clients send turn 2 as turn 1
plus new tokens. Today `prefix-caching` is on in `flashnext.env` and
`27b-awq.env`, and turn 2 should hit.

The stub must not change that. A later real indexer that turns APC off,
or salts the hash, will miss. Measure the miss. Do not paper over it
with a client flag.

Protocol, one model, TP=4, streaming on, c=1:

1. Boot A: `VLLM_LSA` unset, prefix caching on (the recipe). Turn 1 is
   a fixed prompt of 8192 tokens. Turn 2 appends 512 tokens and asks
   for 128 more. Record prefix-cache hit tokens on turn 2, TTFT of
   turn 2, and the first SSE chunk time.
2. Boot B: `VLLM_LSA=1`, same recipe, same two turns, same client body.
   Hit tokens and TTFT must match boot A. If they do not, the stub
   salted or disabled the cache and this PR is wrong.
3. Boot C, only after a real selector exists: prefix caching off,
   selector on. Same two turns. Turn 2 hit tokens should be 0. Record
   TTFT. That delta is the multi-turn tax.
4. Boot D, only after a real selector exists: prefix caching on, block
   hash salted for selector blocks. Same two turns under the salt
   (hit expected within D). Then one request with `VLLM_LSA` unset
   against those blocks (hit must be 0, so an exact request cannot
   reuse selected KV). Then the reverse: exact blocks must not satisfy
   a salted request.

| Boot | Flag | APC | Turn 2 hit tokens | Turn 2 TTFT (s) | Notes |
| --- | --- | --- | --- | --- | --- |
| A | unset | on | | | recipe baseline |
| B | 1 | on | | | must match A |
| C | selector | off | | | empty until the gather exists |
| D | selector | salted | | | empty until the gather exists |
| D, then exact client | unset | on | | | must not reuse D's blocks |

The client script is the same for A–D. Only the server environment
changes.

## MTP

Empty until the selector exists. Boot the 27B recipe with `MTP=2` and
`VLLM_LSA=1`. The stub must serve. Acceptance rate is compared to the
flag unset. A real selector should refuse MTP until draft tokens and
the open cluster block are specified, and this row should then fail
closed instead of reporting a rate.

| Model | MTP | Flag | Result |
| --- | --- | --- | --- |
| Qwen3.8-27B | 2 | 1 | |
| Qwen3.8-27B | 2 | selector | fail closed, not an acceptance rate |

## Graphs

| Mode | Flag | Prefill | Decode | Result |
| --- | --- | --- | --- | --- |
| `FULL_AND_PIECEWISE` | 1 | eager or captured, recipe default | recipe default | logits match flag off |
| `FULL_AND_PIECEWISE` | selector | eager, no graph | one graph per length bucket | empty |
| `--enforce-eager` | selector | eager | eager | empty |

A host readback of list sizes inside a captured prefill is a failure,
not a fallback.
