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
beyond the measured transient peak. At TP=2 with the tower, prompts of 261,052, 942,628 and 1,046,914 tokens
all peaked at 27.20 GiB of 29.98 on both cards (the cache is committed at boot; the figure is the same right
after it): 2.78 GiB stayed free [measured]. A text-only TP=2 engine
with the default floor loses 11,264 tokens (1.07%) against 1,048,576; with the tower it would lose more
[derived from the boots above].

**TP=4** (HIP 2 - 5, `SEQS` 8, `MAXBAT` 4,096, graphs, `--max-model-len 1048576`, the engine's default 3 GiB
floor) [measured]: with the tower loaded the pool holds 2,250,192 tokens (37,552 blocks per rank at the
warmup estimate, 18.34 GiB), 2.15 times the model's length; text only 2,331,686 tokens (2.22x).

Needle runs at the recipes' settings (tower loaded, 1,048,576-token engine, graphs; TP=2: `MAXBAT` 184 and
floor 2 GiB; TP=4: `MAXBAT` 4,096 and the default floor): one cold prefill, a warm repeat, a 96-token decode at
that depth and a 4,088-token control with the same needle, the needle at depth 0.5 [measured]. 1,046,914 tokens
is 99.8% of the model's length:

| TP | Prompt tokens | Prefill (cold) | Warm repeat | Decode at depth | Needle / control | Device peak |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| 1 (auto-fit 180,928) | 162,814 | 251.1 s = 648 tok/s | - | 31.9 ms/token | found / - | 25.59 GiB |
| 2 | 261,052 | 321.8 s = 811 tok/s | - | 24.4 ms/token | found / - | 27.20 GiB |
| 2 | 942,628 | 2,492.9 s (41.5 min) = 378 tok/s | 0.43 s | 49.8 ms/token | found / found | 27.20 GiB |
| 2 | 1,046,914 | 3,000.5 s (50.0 min) = 349 tok/s | 0.31 s | 54.3 ms/token | found / found | 27.20 GiB |
| 4 | 261,052 | 164.3 s = 1,589 tok/s | - | 17.1 ms/token | found / - | 25.58 GiB |
| 4 | 942,628 | 1,266.9 s (21.1 min) = 744 tok/s | 0.38 s | 30.1 ms/token | found / found | 25.58 GiB |
| 4 | 1,046,914 | 1,516.9 s (25.3 min) = 690 tok/s | 0.29 s | 32.0 ms/token | found / found | 25.58 GiB |

Against the earlier eager tree on the same 942,628-token prompt (TP=2 343 tok/s and 51.8 ms/token, TP=4
682 tok/s and 32.1 ms/token) prefill is 10% / 9% faster and decode 4% / 6% faster. The prefill rate falls with
depth, so the 50 and 25 minutes are not extrapolations of the 261k rates. The KFD `evicted_ms` counter read 0 for
every process of the 942,628 and 1,046,914-token runs.

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

## Decode byte levers (opt-in, TP=1)

A b=1 decode step reads 5.3 GB at TP=1, 3.5 GB of it the fp16 attention input packs, `o_proj` and the
`lm_head`; the dense kernels stream those at 410 - 440 GB/s, so bytes are time. Four environment switches
(all OFF by default, not in the recipes) replace them, for steps of up to 4 token rows, by weight-only
int8 (group 64) or int4 (group 32) copies read by native HIP kernels (`inkling2_qlinear_*`, design,
oracles, mutants and measurements in `docs/rdna2/inkling2-qlinear-kernels.md`). The prefill, chunks of
more than 4 rows and TP > 1 keep the fp16 kernels (the flags are ignored at TP > 1).

| Key | Values | Replaces |
| --- | --- | --- |
| `VLLM_INKLING2_ATTN_IN_QUANT` | `w8`, `w4` (`w4g64`, `w8g64` spelled out) | the attention input pack, 24 layers |
| `VLLM_INKLING2_O_PROJ_QUANT` | same | `o_proj`, 24 layers |
| `VLLM_INKLING2_ATTN_QUANT` | same | default of the two above |
| `VLLM_INKLING2_LM_HEAD_QUANT` | same | the `lm_head` (logits and the fused greedy argmax) |
| `VLLM_INKLING2_QUANT_FILE` | path | GPTQ integers and scales for the attention packs (`tools/rdna/inkling2/gptq_quant.py`) instead of round to nearest |

Measured at TP=1 with graphs (`tools/rdna/inkling2/decode_gate.py`: `max_num_batched_tokens` 4, ms per
generated token of one request; the lever-free engine measures 19.05 and 19.10; the gate is the
teacher-forced mean `|dlogp|` against the TP=1 eager dump, limit 0.1027, the lever-free engine scores
0.040):

| Levers | b=1 ms/token | tok/s | NLL gate | Extra weights |
| --- | ---: | ---: | --- | ---: |
| `LM_HEAD_QUANT=w8` | 18.09 | 55.3 | 0.042 pass | +0.44 GiB |
| `LM_HEAD_QUANT=w4` | 17.63 | 56.7 | 0.076 pass | +0.24 GiB |
| `ATTN_QUANT=w8` | 16.27 | 61.5 | 0.056 pass | +0.63 GiB |
| `ATTN_QUANT=w8 LM_HEAD_QUANT=w8` | 15.28 | 65.5 | 0.056 pass | +1.07 GiB |
| `ATTN_QUANT=w8 LM_HEAD_QUANT=w4` | 14.78 | 67.7 | 0.078 pass | +0.87 GiB [derived] |
| `ATTN_QUANT=w4 LM_HEAD_QUANT=w4` (round to nearest) | 13.49 | 74.2 | 0.170 FAIL | +0.59 GiB |
| the same with an offline GPTQ attention file (`VLLM_INKLING2_QUANT_FILE`) | 13.39 | 74.7 | 0.142 FAIL | +0.59 GiB |

These are PROTOTYPES that quantize the weights AT LOAD from the native bf16 -> fp16 weights (about a
second, no per-token cost); the native weights stay the default and are loaded untouched. They are not
a shipping path: a quantized tier has to be an offline checkpoint. The quantized copy lives next to the
fp16 weights, so each lever takes KV cache: all-int8 holds about 35k tokens less context at TP=1
(32 KiB per token) [derived]. Over 3999 teacher-forced positions the all-int8 build shifts the mean
log-probability by -0.0007 +- 0.0016 (lever-free engine against itself +0.0003 +- 0.0014): no
measurable cost. int4 attention does not pass the gate, with round to nearest or with GPTQ (rows in
`docs/rdna2/inkling2-qlinear-kernels.md`, section 7).

## Known limits

- **Speculative decoding (DSpark) is opt-in and not in the recipe defaults.** It speeds one or two requests
  up (TP=1 1.24x / 1.11x, TP=2 1.29x / 1.14x), loses from three, and costs context length (section above);
  the 1,048,576-token TP=2 / TP=4 recipes keep it off. The sampled (non-greedy) draft raises
  `NotImplementedError` at TP > 1; TP=4 was not run with the draft.
- **One image per prompt** in the recipes (`--limit-mm-per-prompt {"image":1}`); the Inkling processor takes
  no `--mm-processor-kwargs`, so the launcher's `vision-cap` feature is not used.
- **Graphs cover decode steps of up to `min(16, SEQS)` requests and mixed steps of up to 32 tokens.**
  A prefill chunk runs eager. A request count of the decode grid replays in its own graph.
- **A full-length prompt is a long job**: 1,046,914 tokens (99.8% of 1,048,576) prefill in 50.0 minutes at TP=2
  (349 tok/s) and 25.3 minutes at TP=4 (690 tok/s), against 811 and 1,589 tok/s for a 261k prompt (the rate
  falls with depth). Each was run once, one needle at depth 0.5; several long prompts at once and a long prompt
  with an image were not run.
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
256 KiB. TP=1 (on HIP 4) and TP=2 are the rebuilt merged tree (0722fa5075); TP=4 is the same code one build
earlier (the merge added the DSpark kernel, which this path does not call).

| requests | TP=1 eager | TP=1 graph | ratio | TP=2 eager | TP=2 graph | ratio | TP=4 eager | TP=4 graph | ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 20.37 | 19.32 | 0.948 | 15.98 | 14.12 | 0.883 | 16.94 | 11.54 | 0.681 |
| 2 | 24.03 | 23.22 | 0.966 | 17.45 | 16.20 | 0.928 | 17.04 | 12.65 | 0.742 |
| 3 | 27.69 | 26.73 | 0.965 | 19.41 | 18.18 | 0.937 | 18.77 | 13.76 | 0.733 |
| 4 | 30.96 | 30.00 | 0.969 | 21.23 | 19.99 | 0.942 | 17.47 | 15.00 | 0.859 |
| 5 | 33.10 | 32.27 | 0.975 | 22.66 | 21.17 | 0.935 | 16.92 | 15.54 | 0.918 |
| 6 | 35.51 | 34.62 | 0.975 | 23.76 | 22.57 | 0.950 | 18.47 | 17.02 | 0.922 |
| 7 | 38.47 | 37.77 | 0.982 | 25.85 | 24.55 | 0.950 | 20.00 | 18.60 | 0.930 |
| 8 | 40.72 | 39.91 | 0.980 | 27.08 | 25.94 | 0.958 | 20.69 | 19.10 | 0.923 |
| 12 | 48.17 | 47.67 | 0.990 | 31.70 | 30.42 | 0.960 | 24.38 | 22.90 | 0.939 |
| 16 | 50.31 | 48.22 | 0.958 | 32.82 | 31.69 | 0.966 | 25.58 | 24.55 | 0.960 |

Graph throughput (all requests together, tokens per second): 1 request TP=1 51.8, TP=2 70.8, TP=4 86.7;
8 requests 200.5 / 308.4 / 418.8; 16 requests 331.8 / 504.8 / 651.6. The TP=4 eager column is noisy (3
requests are slower than 4); the large TP=4 gain at 3 requests or fewer (ratios 0.68 - 0.74) is probably the
host launch cost the graph removes [inference].

### Boot, capture and prefix cache

| | TP=1 | TP=2 | TP=4 |
| --- | ---: | ---: | ---: |
| engine boot, eager / graphs (`graph_check`, 4,096-token engine) | 24.7 / 25.8 s | 34.7 / 36.6 s | 36.9 / 39.1 s |
| graph capture | 2 s, 0.70 GiB | 2 s | 2 s (the log's memory counter reads -0.20 GiB) |
| 1,301-token prompt, cold / warm (graphs) | 1.54 / 0.39 s | 0.97 / 0.28 s | 0.70 / 0.23 s |

### Correctness with graphs

Teacher-forced NLL of 799 positions against the TP=1 eager run A (limit: 2 x the A-vs-B floor of 0.0463 + 0.01 =
0.1027): TP=1 graphs 0.0463, TP=2 graphs 0.0450, TP=4 graphs 0.0446 (eager: TP=2 0.0437, TP=4 0.0416); these
NLL runs predate the rebuild. Greedy continuations of 30 prompts identical to eager: TP=1 17 on the rebuilt tree
(20 in the first run; the eager-against-eager floor is 21), TP=2 22 (25), TP=4 21; every first divergence is at a
near-tie (largest gap 0.22 nats at TP=1, 0.11 at TP=2, 0.23 at TP=4). Details: port spec, "merged tree: final
results".

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

Final library before the merge, HIP 3: kernel suites 863 + 1,075 passed; `test_serving_path_gpu.py` +
`test_serving_path_graph_gpu.py` 146 passed; the ring guards killed 3 of 3 mutants (c9 - c11 of
`inkling2-attn-kernels.md`); CPU suite of the model, the recipes and the margin floor 686 passed, 97
skipped, 0 failed. The merged tree was rebuilt (131 s, exit 0); the decode tables and the full-length runs above
ran on it, the suites were not re-run on the rebuilt library. Pre-commit clean.

## Related notes

- [`inkling2-port-spec.md`](inkling2-port-spec.md): architecture, the engineering record and the merged-tree
  final results.
- [`inkling2-attn-kernels.md`](inkling2-attn-kernels.md): the attention and conv-ring kernels and their tests.
- [`inkling2-tensor-parallel.md`](inkling2-tensor-parallel.md): the TP=2 / TP=4 layout.
- [`inkling2-vision.md`](inkling2-vision.md): the vision tower and its kernels.
- [`inkling2-dspark.md`](inkling2-dspark.md): the draft model, its kernels, the K sweep, TP=2, equality
  and the memory it costs.
