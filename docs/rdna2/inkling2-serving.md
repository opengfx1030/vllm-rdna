# Inkling-2-Mini on V620: serving at TP=1, TP=2 and TP=4 with HIP graphs

Inkling-2-Mini (`thinkingmachines/Inkling-2-Mini`, architecture
`Inkling2LoopForConditionalGeneration`) is a looped 24-layer mixture-of-experts model (hidden
2304, 128 experts with top-8 routing in every layer, NVFP4 experts in layers 0-19 and MXFP8
in 20-23, no RoPE) with a vision tower and a context of 1,048,576 tokens. gfx1030 cards (V620,
29.98 GiB each) serve it with native HIP kernels only: no rocBLAS, no Triton, no torch
compute in the serving path. The engine is this fork's V2 model runner with the
`FULL_AND_PIECEWISE` HIP-graph mode, prefix caching and the vision input.

This page is the user guide: how to serve, what the switches cost, the memory and length
limits, and the measurements behind the three recipes. The architecture, the kernels and the
engineering record are in [`inkling2-port-spec.md`](inkling2-port-spec.md) (section "merged
tree: final results"), [`inkling2-attn-kernels.md`](inkling2-attn-kernels.md),
[`inkling2-glue-kernels.md`](inkling2-glue-kernels.md),
[`inkling2-linear-kernels.md`](inkling2-linear-kernels.md),
[`inkling2-tensor-parallel.md`](inkling2-tensor-parallel.md) and
[`inkling2-vision.md`](inkling2-vision.md). Labels: **[measured]** = a run listed in
"Measured results", **[derived]** = arithmetic on measured values, **[inference]** = a run must
confirm it.

## Serve

From the repository root. `MODEL` is the checkpoint directory, `VENV` a ROCm 7.14 venv that
has this tree's extensions built, `HIP_VISIBLE_DEVICES` the cards (their order is the TP
rank order).

```bash
bash tools/rdna/serve_rdna.sh RECIPE=inkling2-mini-tp1 MODEL=/path/to/inkling2-mini \
  VENV=/path/to/venv HIP_VISIBLE_DEVICES=3
```

| Recipe | Cards | Port | `MAXLEN` | `SEQS` | `MAXBAT` | Use it for |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| [`inkling2-mini-tp1`](../../tools/rdna/recipes/inkling2-mini-tp1.env) | 1 | 18221 | 180,000 | 16 | 1,024 | one card: short and medium contexts, up to 16 concurrent requests |
| [`inkling2-mini-tp2`](../../tools/rdna/recipes/inkling2-mini-tp2.env) | 2 | 18222 | 1,048,576 | 8 | 184 | the full context on two cards; lower latency than TP=1 |
| [`inkling2-mini-tp4`](../../tools/rdna/recipes/inkling2-mini-tp4.env) | 4 | 18223 | 1,048,576 | 8 | 4,096 | the full context with room for two such requests; fastest prefill |

All three serve the model as `inkling2-mini` on `127.0.0.1`, capture HIP graphs
(`FULL_AND_PIECEWISE`, the model's default capture grid), enable prefix caching and accept one
image per prompt. `PRINT=1` prints the environment and the exact command without launching.
A text request and an image request (the image as a data URL):

```bash
curl -s http://127.0.0.1:18221/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"inkling2-mini","messages":[{"role":"user","content":"What is the capital of France?"}],"max_tokens":64}'

IMG=$(base64 -w0 picture.png)
curl -s http://127.0.0.1:18221/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"inkling2-mini","max_tokens":64,"messages":[{"role":"user","content":[
        {"type":"image_url","image_url":{"url":"data:image/png;base64,'"$IMG"'"}},
        {"type":"text","text":"Describe the image in one sentence."}]}]}'
```

The model thinks before it answers: the reply starts with `<|content_thinking|>` text, so a
small `max_tokens` ends inside the thinking. `python tools/rdna/inkling2/serve_smoke.py --url
http://127.0.0.1:18221` runs a text, an image, a prefix-cache, a streaming and a concurrent
request against a running server and prints `RESULT PASS` or `RESULT FAIL`.

Precedence, lowest to highest: recipe file, variables already in the environment, trailing
`KEY=value` arguments (so every key below can be changed on the command line).

What a good boot logs:

- `rdna_ar: one-shot all-reduce active (handle 0, rank 0/2, ... up to 256 KB, RCCL above ...)`
  at TP > 1 (two handles: the fp32 all-reduce after the attention output projection and the
  fp16 one after the MoE).
- `Model loading took ... GiB`, then `Graph capturing finished in 1 - 2 secs, took 0.2 - 0.7 GiB`.
- `GPU KV cache size: N tokens, Maximum concurrency for M tokens per request: Kx` with `K` at
  least 1.00.
- At TP=2: `VLLM_KV_CACHE_MARGIN_FLOOR_MIB=2048: the KV cache sizing keeps 2.0 GiB of device
  memory free beyond the measured transient peak instead of 3.0 GiB.`

Ready (the `/v1/models` endpoint answers, polled every 2 s from the launch, weights loading and graph
capture included): TP=1 44 s, TP=2 64 s, TP=4 59 s [measured, `serve_one.sh`].

## Switches

| Key | Recipe value | Off / other | What it does | Cost, measured |
| --- | --- | --- | --- | --- |
| `EAGER` | 0 | 1 | `1` runs without HIP graphs (`--enforce-eager`) | decode +2 to +13% per token (table below), KV pool +10% at TP=1, SEQS 16 |
| `SEQS` | 16 (TP=1), 8 | 1 - 16 | `--max-num-seqs`; decode graphs for 1 .. min(16, `SEQS`) requests | each size is a graph; at TP=1 the 16 + 7 graphs take 0.70 GiB of the KV budget |
| `MAXBAT` | 1,024 / 184 / 4,096 | any | `--max-num-batched-tokens`: the prefill chunk; steps above 32 tokens run eager | at TP=2 every token of it takes 1.92 MiB of the KV budget [derived from the boots below]; larger chunks prefill faster |
| `MAXLEN` | 180,000 / 1,048,576 | lower | `--max-model-len`; the engine refuses a length the KV pool cannot hold once | see "Long context" |
| `VLLM_KV_CACHE_MARGIN_FLOOR_MIB` | 2048 (TP=2 only) | empty | the memory the KV sizing keeps free beyond its measured transient peak (engine default 3 GiB on ROCm) | the full context at TP=2 needs it (see "Long context") |
| `RDNA_AR`, `VLLM_RDNA_AR_MAX_KB`, `VLLM_RDNA_AR_ONESHOT_KB` | 1, 256, 256 (TP > 1) | `RDNA_AR=0` | the one-shot HIP all-reduce up to 256 KiB (28 decode tokens in fp32), RCCL above; captured inside the graphs | the cost of RCCL for every reduction was not measured: no decode step needed the fallback |
| `COMPILATION_CONFIG` | unset | JSON | an explicit capture grid, e.g. `{"cudagraph_mode":"FULL_AND_PIECEWISE","decode_cudagraph_capture_request_sizes":[1,2,4,8]}` | fewer graphs, less KV budget spent; a request count without its own graph replays in the next larger one (3 rows in the 4-row graph cost 35.4 ms against 30.8 ms in an exact one, port spec part B) |
| `EXTRA_ARGS='--limit-mm-per-prompt {"image":1}'` | one image per prompt | `{"image":0}` | `0` does not load the vision tower | saves 0.6 GiB per rank at TP=2 (weights 9.0 against 9.6 GiB) |

The default capture grid for this architecture is decode graphs for every request count from 1
to min(16, `SEQS`) and mixed / prefill graphs for 1, 2, 4, 8, 16, 24 and 32 tokens; a step with
more tokens (a prefill chunk) runs eager.

## Slots and memory

Per rank, from the engine's boot log of the recipes [measured]:

| | TP=1 | TP=2 | TP=4 |
| --- | ---: | ---: | ---: |
| weights ("Model loading took"), vision tower loaded | 18.68 GiB | 9.6 GiB | 6.21 GiB |
| weights, text only (`{"image":0}`) | - | 9.0 GiB | 5.61 GiB |
| KV per token (the 8 full-attention layer visits of the loop; the sliding-window layers add a bounded amount per request) | 32 KiB | 16 KiB | 8 KiB |
| one 1,048,576-token request | 32 GiB | 16.1 GiB | 8.2 GiB |
| HIP graphs (engine log "took") | 0.70 GiB (SEQS 16) | 0.19 GiB (SEQS 8) | between -0.50 and 0.30 GiB (below the counter's resolution) |
| KV pool of the recipe | 180,986 tokens | 1,062,736 tokens | 2,250,192 tokens |

The pool is what is left: free memory after warmup and graph capture, less the margin (engine
default 3 GiB on ROCm, 2 GiB at TP=2 in the recipe) and the transient activation peak the warmup measured
(0.77 GiB at TP=1, 0.3 GiB at TP=2 with chunks of 184 tokens). Capture comes out of the same budget: at
TP=1 SEQS 16 costs 9.5% of the KV tokens of a 4,096-token engine (45,175 eager, 40,876 with graphs).
Prefix caching keeps conv state per 64-token block, so a hit covers whole blocks (128 of 134 tokens in
the image test below).

## Long context

**TP=1** (HIP 3, tower loaded, `MAXBAT` 1,024, `--max-model-len -1` lets the engine fit the length after
capture and warmup) [measured]:

| Graphs | `SEQS` | Largest length that fits | KV blocks (GiB) |
| --- | ---: | ---: | ---: |
| `FULL_AND_PIECEWISE` (recipe) | 16 | 180,928 | 3,107 (6.07) |
| `FULL_AND_PIECEWISE` | 2 | 194,368 | - |
| eager (merged tree, earlier run) | 2 | 201,600 | - |

A 162,814-token prompt (90% of the 180,928-token fit, `--max-num-seqs 16`, graphs) prefilled in 251.1 s
(648 tok/s, eager 475 tok/s for 181,434 tokens on the earlier tree), decoded at 31.9 ms/token at that depth
(eager 35.5 ms at 181k) and the needle at depth 0.5 was found; the device peaked at 25.59 GiB of 29.98
[measured].

**TP=2** (HIP 4,5, `SEQS` 8, `--max-model-len 1048576`, graphs) [measured]. The engine refuses a length the
KV cache cannot hold once; its message gives the estimate:

| Tower | `MAXBAT` | Margin floor | Blocks (GiB) | Needed | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| none | 184 | 3 GiB (default) | 16,338 (15.95) | 16.13 GiB | refused, estimated maximum 1,037,312 |
| none | 160 | 3 GiB | 16,356 (15.97) | 16.12 GiB | refused, 1,038,784 |
| none | 96 | 3 GiB | 16,444 (16.06) | 16.11 GiB | refused, 1,045,312 |
| none | 88 | 3 GiB | 16,472 (16.08) | 16.11 GiB | refused, 1,047,232 |
| none | 184 | 2 GiB | 17,362 (16.96) | 16.1 GiB | boots, 1,102,487 tokens, 1.05x |
| loaded | 184 | 2 GiB (recipe) | 16,736 (16.34) | 16.1 GiB | boots, 1,062,736 tokens, 1.01x |
| none, eager (merged tree, earlier run) | 184 | 3 GiB | 16,530 | 16.13 GiB | boots |

With the default floor and graphs the engine's estimate at 88 tokens per step is still 1,344 tokens short, so
the full context would need a chunk of roughly 70 - 80 tokens [INFERENCE from the slope of the four boots, not
booted]: a step that streams the whole expert set for so few tokens is not a way to serve a million-token
prompt, so the recipe lowers the floor. `VLLM_KV_CACHE_MARGIN_FLOOR_MIB=2048`
gives the cache the 1 GiB difference; what was lost against the default is 1 GiB of free device memory
beyond the measured transient peak. At TP=2 with the tower the 262,144-token needle (the prompt is 261,052
tokens) peaked at 27.20 GiB of 29.98 on both cards: 2.78 GiB stayed free [measured]. A text-only TP=2 engine
with the default floor loses 11,264 tokens (1.07%) against 1,048,576; with the tower it would lose more
[derived from the boots above].

The 261,052-token needle prompt (tower loaded, 1,048,576-token engine, graphs, `MAXBAT` 184, floor 2 GiB)
prefilled in 321.8 s (811 tok/s; 665 tok/s eager on the earlier tree), decoded at 24.4 ms/token at that depth
(eager 26.4 ms) and the needle at depth 0.5 was found [measured].

**TP=4** (HIP 2 - 5, `SEQS` 8, `MAXBAT` 4,096, graphs, `--max-model-len 1048576`, the engine's default 3 GiB
floor) [measured]: with the tower loaded the pool holds 2,250,192 tokens (37,552 blocks per rank at the
warmup estimate, 18.34 GiB), 2.15 times the model's length; text only 2,331,686 tokens (2.22x). The
261,052-token needle (tower loaded) prefilled in 164.3 s (1,589 tok/s), decoded at 17.1 ms/token at that
depth and was found at depth 0.5; the device peaked at 25.58 GiB of 29.98 (HIP 2).

## Speculative decoding (DSpark, opt-in)

The model's own draft (`mtp.safetensors`: three layers, a block of 8 rows, a Markov head) runs on native
HIP kernels under the same `FULL_AND_PIECEWISE` graphs, with prefix caching, at TP=1 and TP=2; TP=4 shares
the TP=2 code path and was not run through the engine. **It is not in the recipe defaults**: at this
model's MoE row cost it pays at one or two requests only, and it takes context length. Switch it on with
the launcher key `SPEC_CONFIG`; the model requires `num_speculative_tokens` 7, and the schedule decides
how many of the 7 drafted tokens the engine verifies per step (2 for one or two requests, none above):

```bash
bash tools/rdna/serve_rdna.sh RECIPE=inkling2-mini-tp1 MODEL=/path/to/inkling2-mini VENV=/path/to/venv \
  HIP_VISIBLE_DEVICES=3 SEQS=2 MAXLEN=125000 \
  SPEC_CONFIG='{"method":"dspark","num_speculative_tokens":7,"num_speculative_tokens_per_batch_size":[[1,2,2]]}'
```

TP=2: `RECIPE=inkling2-mini-tp2 HIP_VISIBLE_DEVICES=4,5 SEQS=2 MAXLEN=780000`. With more slots keep the
tail of the schedule: `[[1,2,2],[3,8,0]]`.

Decode ms per token of one request (graph replay, 24 chat prompts x 128 tokens, speed-up over no draft in
brackets; the TP=2 rows are one session on the same two cards):

| | b1 | b2 | b3 | b4 | b8 |
|---|---|---|---|---|---|
| TP=1 no draft | 19.11 | 23.40 | 27.38 | 31.11 | 43.22 |
| TP=1 K=2 | 15.43 (1.24x) | 21.09 (1.11x) | 28.69 (0.95x) | 32.81 (0.95x) | 41.16 (1.05x) |
| TP=1 K=3 | 15.63 (1.22x) | 21.43 (1.09x) | 29.14 (0.94x) | 32.32 (0.96x) | 42.72 (1.01x) |
| TP=2 no draft | 14.03 | 16.17 | - | - | - |
| TP=2 K=2 | 10.90 (1.29x) | 14.15 (1.14x) | - | - | - |
| TP=2 K=3 | 10.68 (1.31x) | 14.45 (1.12x) | - | - | - |

An earlier TP=2 session with 8 slots: no draft 14.12 / 16.32 / 20.63 / 28.54 at b = 1 / 2 / 4 / 8, K=3
10.90 / 14.98 / 23.03 / 29.19 (1.30x, 1.09x, 0.90x, 0.98x). The greedy output equals the no-draft output
within the run-to-run floor (teacher-forced gaps at TP=1 and TP=2; cold, hit and duplicate prefix-cache
requests included): `docs/rdna2/inkling2-dspark.md`.

The TP=1 rows were measured in a 4,096-token, 8-slot engine. Booted with the recipe's own settings (SEQS 2,
125,000 tokens, vision tower, K=2) the same prompts run at 15.65 / 21.28 ms at b1 / b2 (1.22x / 1.10x); the
TP=2 K=2 row is the recipe engine (SEQS 2, 780,000 tokens).

What the draft costs (the engine's own KV planner with the recipe flags; longest context in tokens):

| | no draft | with the draft |
|---|---|---|
| TP=1, SEQS 2 | 194,368 | 125,824 with K=2 (125,000 booted), 131,392 with K=3 |
| TP=1, SEQS 8 | - | 96,512 |
| TP=1, SEQS 16 | 180,986 | 51,072 |
| TP=2, SEQS 8 | 1,062,736 (1,048,576 fit) | 737,088 |
| TP=2, SEQS 2 | 1,073,339 | about 790,000 (780,000 booted) |

The draft adds 0.8 - 0.9 GiB of weights per rank, 4 KiB of KV per token and rank (its full-attention layer
is replicated at TP > 1) and graph memory that grows with SEQS (0.68 GiB at SEQS 2 to 3.44 GiB at SEQS 16 at
TP=1, against 0.70 GiB without the draft). A window on the draft's full-attention layer would give the
full context back at TP=2; it is not built (`docs/rdna2/inkling2-dspark.md`).

## Known limits

- **Speculative decoding (DSpark) is opt-in and not in the recipe defaults.** It speeds one or two requests
  up (TP=1 1.24x / 1.11x, TP=2 1.29x / 1.14x), loses from three, and costs context length (section above);
  the 1,048,576-token TP=2 / TP=4 recipes keep it off. The sampled (non-greedy) draft raises
  `NotImplementedError` at TP > 1; TP=4 was not run with the draft.
- **One image per prompt** in the recipes (`--limit-mm-per-prompt {"image":1}`); the Inkling processor takes
  no `--mm-processor-kwargs`, so the launcher's `vision-cap` feature is not used.
- **Graphs cover decode steps of up to `min(16, SEQS)` requests and mixed steps of up to 32 tokens.**
  A prefill chunk runs eager. A request count of the decode grid replays in its own graph.
- **A full 1,048,576-token prompt was not run.** Its prefill is about 51 minutes at TP=2 at the eager rate
  of the earlier tree (942,628 tokens at 343 tok/s average [derived]); the 262k needle prefilled 22% faster
  here, so about 42 minutes [inference]. TP=4: about 28 minutes at the earlier rate [inference]. The needles
  of this page are 262,144 tokens (TP=1: 90% of the fit).
- **TP=1 holds about 181k tokens, not the model's million**: the KV cache is 32 KiB per token (eight
  full-attention layer visits, counting the three passes of the loop) and 18.68 GiB of the 29.98 GiB are
  weights. The recipe is 180,000.
- **The TP=2 recipe lowers the engine's KV margin floor to 2 GiB.** With the default 3 GiB the full context
  does not fit once graphs are captured (see "Long context").
- **Greedy output is not bit-stable**: two runs of the same eager engine produce different continuations
  for a third of the prompts, at near-ties of the logits. Graph replay is within that noise. Judge a change
  by the teacher-forced NLL, not by the identity of the generations.
- **`tools/rdna/inkling2/real_check.py` prints `PREFIX HIT FAIL` and a first-step limit failure**; both are
  known false failures of the check (it compares two noisy numbers).

## Measured results

Real checkpoint, merged tree built on the box, V620 cards shared with other jobs (timings indicative;
`evicted_ms` monitored). Tools: `tools/rdna/inkling2/graph_check.py`, `long_context_check.py`,
`real_check.py`, `vision_engine_check.py`, `serve_smoke.py`.

### Decode, ms per generated token of one request (= the step time of that many rows)

`graph_check.py --max-model-len 4096 --max-num-seqs 16 --prefix-caching --tokens 64 --reps 3`, eager
(`--enforce-eager`) against `FULL_AND_PIECEWISE` on the default grid; TP > 1 on the one-shot all-reduce up to
256 KiB. TP=1 and TP=2 come from the build before the last two ring guards (one compare per ring write, not
on the decode critical path [inference]), TP=4 from the final library.

| requests | TP=1 eager | TP=1 graph | ratio | TP=2 eager | TP=2 graph | ratio | TP=4 eager | TP=4 graph | ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 20.09 | 19.12 | 0.952 | 16.21 | 14.16 | 0.873 | 16.94 | 11.54 | 0.681 |
| 2 | 23.78 | 23.28 | 0.979 | 17.55 | 16.04 | 0.914 | 17.04 | 12.65 | 0.742 |
| 3 | 27.20 | 26.53 | 0.976 | 19.35 | 18.00 | 0.930 | 18.77 | 13.76 | 0.733 |
| 4 | 30.70 | 30.12 | 0.981 | 21.26 | 20.04 | 0.942 | 17.47 | 15.00 | 0.859 |
| 5 | 32.87 | 32.08 | 0.976 | 22.46 | 21.32 | 0.949 | 16.92 | 15.54 | 0.918 |
| 6 | 35.15 | 34.40 | 0.979 | 23.80 | 22.63 | 0.951 | 18.47 | 17.02 | 0.922 |
| 7 | 38.42 | 37.29 | 0.971 | 25.80 | 24.50 | 0.950 | 20.00 | 18.60 | 0.930 |
| 8 | 40.40 | 39.62 | 0.981 | 26.94 | 25.84 | 0.959 | 20.69 | 19.10 | 0.923 |
| 12 | 47.73 | 46.65 | 0.977 | 31.48 | 30.30 | 0.962 | 24.38 | 22.90 | 0.939 |
| 16 | 50.57 | 48.69 | 0.963 | 33.82 | 33.84 | 1.001 | 25.58 | 24.55 | 0.960 |

Graph throughput (all requests together, tokens per second): 1 request TP=1 52.3, TP=2 70.6, TP=4 86.7;
8 requests 201.9 / 309.6 / 418.8; 16 requests 328.6 / 472.8 / 651.6. The TP=4 eager column is noisy (3
requests are slower than 4); the large TP=4 gain at 3 requests or fewer (ratios 0.68 - 0.74) is probably the
host launch cost the graph removes [inference].

### Boot, capture and prefix cache

| | TP=1 | TP=2 | TP=4 |
| --- | ---: | ---: | ---: |
| engine boot, eager / graphs (`graph_check`, 4,096-token engine) | 37.3 / 39.1 s | 35.4 / 37.7 s | 36.9 / 39.1 s |
| graph capture | 2 s, 0.70 GiB | 2 s | 2 s (the log's memory counter reads -0.20 GiB) |
| 1,301-token prompt, cold / warm (graphs) | 1.53 / 0.39 s | 0.97 / 0.28 s | 0.70 / 0.23 s |

### Correctness with graphs

Teacher-forced NLL of 799 positions against the TP=1 eager run A (limit: 2 x the A-vs-B floor of 0.0463 + 0.01 =
0.1027): TP=1 graphs 0.0463, TP=2 graphs 0.0450, TP=4 graphs 0.0446 (eager: TP=2 0.0437, TP=4 0.0416). Greedy
continuations of 30 prompts identical to eager: 20 at TP=1 (the eager-against-eager floor is 21), 25 at TP=2,
21 at TP=4; every first divergence is at a near-tie (gap at most 0.25 nats). Details: port spec, "merged tree:
final results".

### Vision (images through the V2 runner with graphs)

`vision_engine_check.py` (a red disc on blue, a green square on yellow, a grey triangle on yellow): one image,
the same image twice, two images in one batch, two images plus a text prompt. TP=1 graphs: `RESULT PASS`;
TP=2 and TP=4 graphs at 1,048,576 tokens: `RESULT PASS` each. The repeated image is a prefix hit (128 of 134
prompt tokens cached, same ids). Answers equal the eager ones except the three-request batch, which differs
from its single answer after token 18 - 41 as it does under eager.

### Served through the launcher (`serve_smoke.py`)

| Recipe | Ready | `max_model_len` | text | image | prefix cold / warm (1,736 tokens) | stream | 4 concurrent | Result |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | --- |
| `inkling2-mini-tp1` | 44 s | 180,000 | Paris | red, blue | 1.70 / 0.19 s | 54.1 chunks/s | 138.8 tok/s | PASS |
| `inkling2-mini-tp2` | 64 s | 1,048,576 | Paris | red, blue | 1.36 / 0.15 s | 72.9 chunks/s | 194.4 tok/s | PASS |
| `inkling2-mini-tp4` | 59 s | 1,048,576 | Paris | red, blue | 0.73 / 0.12 s | 89.5 chunks/s | 245.8 tok/s | PASS |

### Gates

Final library, HIP 3: kernel suites 863 + 1,075 passed; `test_serving_path_gpu.py` +
`test_serving_path_graph_gpu.py` 146 passed; the ring guards killed 3 of 3 mutants (c9 - c11 of
`inkling2-attn-kernels.md`); CPU suite of the model, the recipes and the margin floor 686 passed, 97
skipped, 0 failed. Pre-commit clean.

## Related notes

- [`inkling2-port-spec.md`](inkling2-port-spec.md): architecture, the engineering record and the merged-tree
  final results.
- [`inkling2-attn-kernels.md`](inkling2-attn-kernels.md): the attention and conv-ring kernels and their tests.
- [`inkling2-tensor-parallel.md`](inkling2-tensor-parallel.md): the TP=2 / TP=4 layout.
- [`inkling2-vision.md`](inkling2-vision.md): the vision tower and its kernels.
- [`inkling2-dspark.md`](inkling2-dspark.md): the draft model, its kernels, the K sweep, TP=2, equality
  and the memory it costs.
