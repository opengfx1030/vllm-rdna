# Explore: RidgeFill on this fork

**Status: explore draft, measured on the V620 ([TESTPLAN.md](TESTPLAN.md),
[QUALITY.md](QUALITY.md)). The int8 fill cuts 16K TTFT by 10.2 % against
exact and matches the stock attention path on the same maps, but no map set
meets the +0.03 long-window PPL gate yet: the paper's released maps cost
+0.266 PPL, +0.047 with a 12288-token exact tail; the PG-19 fit is next.**
Nothing here is a serving default.
`VLLM_RDNA_RIDGEFILL` defaults off. With the flag unset, Flash-Next still runs
the exact model: the added branches are `None` checks (model forward and
runner state), and the prefix-hash hook returns before it imports this
package. No recipe, ROCm pin, or torch
pin changes. One new HIP kernel, opt-in and off by default: the fused map
GEMM (`csrc/rocm/ridgefill_map_rdna2.cu`). The W4A16, W4A8, EXL3, and
`rdna_allreduce` sources are untouched.

This is a handoff for people with V620 GPUs. The math, the pin check, and
the flag-off path run on CPU. A flag-on boot without a map path only
installs the guards below and stays exact. With a map path, filled rows
skip the late MoE and write their late-layer cache from the map's
prediction. Each request is cut at its prefill length minus the tail,
rounded down in cache-only mode to KV block x indexer compress ratio (4096
tokens on the `flashnext` recipe), from lengths the model runner state
hands in, because chunked attention metadata ends at the chunk. Filled
forwards run eagerly, because the compiled graph is reused
without guards (TESTPLAN.md, Measured cells).

| Piece | Where |
| --- | --- |
| This design | `README.md` |
| gfx1030 apply path | [SILICON.md](SILICON.md) |
| Scheduler, MTP, prefix cache, CPU KV | [ENGINE.md](ENGINE.md) |
| V620 gates and measured cells | [TESTPLAN.md](TESTPLAN.md) |
| fp32 fit and range check | `reference_ridge.py` |
| Capture and refit | `example_capture_fit.py` |
| One-shot serve env | `example_flashnext.env` |
| Map GEMM kernels, opt-in (fp16 dot2, int8 dot4) | `csrc/rocm/ridgefill_map_rdna2.cu`, `.cuh` |
| Map kernel micro-bench, dot rates, int8 study | `kbench/` |
| Long-window PPL (PG-19, 64 x 16K) | `ppl_long.py` |
| Draft package | `vllm/model_executor/layers/ridgefill/` |

## How to pick this up on a V620

1. Read [SILICON.md](SILICON.md) and [ENGINE.md](ENGINE.md). Path A is the
   one to build: dequant the int8 maps to fp16 at load and apply them with
   the existing fp16 GEMM (fp32 accumulate). bf16 is the overflow fallback.
2. Do not pull `tcclaviger/vllm:29.08.10` onto a gfx1030 box. That image and
   its MXFP4/FP8 checkpoints target RDNA4 (R9700, gfx1201). gfx1030 has no
   FP8 unit. The image's vLLM patches are not public. This draft does not
   copy them.
3. Refit. The released maps are pinned to an MXFP4/FP8 checkpoint. Run
   `example_capture_fit.py` inside the dest torch 2.12 / ROCm 7.14 tree,
   capture in fp32, solve on the CPU. The loader refuses a vocab or encoding
   mismatch.
4. Serve with the example env on top of `RECIPE=flashnext`, `MTP=0`,
   `ENABLE_PREFIX_CACHING=0`. Clients set nothing. The OpenAI API is
   unchanged. Measure TTFT from streaming responses.
5. Fill [TESTPLAN.md](TESTPLAN.md). Do not treat the paper's 1.50–1.60× as a
   V620 forecast: that measurement included experts living in host memory on
   two R9700s.

```bash
python3 tests/model_executor/test_ridgefill.py
python3 docs/explore/ridgefill/reference_ridge.py
python3 docs/explore/ridgefill/example_capture_fit.py --show-pin-failure
```

## What RidgeFill is

Training-free prefill shortening. Layers `0 .. split-1` run exactly. For
prompt rows outside the tail (the last T tokens, 2048 in the Flash-Next
config) the split-layer state predicts each later layer's input,

```text
x_hat_l = A_l x_S + b_l
```

with `A_l`, `b_l` from one closed-form ridge fit. The late layer writes its
cache from that prediction and skips the MLP/MoE. Decode is exact. The tail
is exact, so it can attend to approximate keys.

Paper figures, quoted, not measured here: Qwen3.8-Flash-Next in vLLM,
1.50–1.60× TTFT at 9K–32K, and a 64K request (prefill plus a short answer,
across server starts) 82.6 s to 47.5 s. RULER needles 99.8 against 100.
Quality mode, terminal-state correction plus protected rows, is 1.22–1.27×
at 9K–32K. The paired perplexity gain over plain RidgeFill includes zero.
Source: the paper, section 5 and Table 16.

## Checked against the paper and this tree

| Claim | Result |
| --- | --- |
| Flash-Next split 24/48, tail 2048. Above the split: 6 QSA and 18 GDN. | Paper section 5.2. The release `layer_types` repeats three linear layers and one full-attention layer; full attention above 24 is layers 27, 31, 35, 39, 43, 47. |
| 24 int8 maps, input 10240, output 2560, about 0.60 GiB. | Paper section 2.2. `code/c5/convert.py` at tag v1.8 (`H=2560`, `HC=4`, group 128, layers 24..47). |
| QSA filled rows also write indexer keys. | Paper section 5.2. |
| GDN filled rows still run projections and the recurrent update; the MoE is the skip; the error enters the shared state. | Paper sections 2.1 and 2.3. Dest's fused GDN path accepts fp32 or bf16 state (`FUSED_GDN_STATE_DTYPES` in `qwen_gdn_linear_attn.py`). "Always fp32" is not a blanket fact. |
| Quality mode is optional and about 1.25×. | Table 16: 1.25×, 1.22×, 1.27× at 9K, 16K, 32K. |
| Released maps are fitted on the served MXFP4/FP8 activations. `ridgefill.json` pins `model.vocab_sha256` and `model.encodings`. | Paper Appendix G, and the Flash-Next manifest on Hugging Face. Our W4A16, EXL3, and HI2 checkpoints need a refit. The loader fails closed on a mismatch. |
| Their vLLM patches are not public. Public artifacts are the image, `code/c5/`, and `code/engine-fix/patch-qsa-v3.diff`. | Paper Appendix F: "the patches themselves are not public." This box has no Docker, so the image was not opened. |
| 27B maps are for the llama.cpp GGUF, split 16, value rescale `c=512`, about 1.19–1.31×. | Paper section 4 and the 27B model card. Second target, after a refit. |
| Qwen3.5-9B: the fitted map beats identity by 3.3 RULER points, interval through zero. | Paper Table, reference study. Leave it. |

The indexer determinism patch (`code/engine-fix/`) sorts block selection and
replaces the fused top-k above 2048 columns. The paper says it costs 5–7%
decode and that every measured arm ran with it. Dest's QSA path
(`vllm/models/qwen4_exp/amd/ops/qsa.py`) does not contain that patch. Do not
port it until two exact runs on one boot disagree. Default: leave it out.

## Recommended path

1. **Flash-Next first, plain mode, tail 2048, split 24.** No protected rows,
   no terminal-state correction, no two-sweep scheduler.
2. **Refit** on the checkpoint we serve. `capture_activations.py --model ...
   --stats --corpus-json FIT --held-json HELD` captures in the dest torch
   2.12 / ROCm 7.14 image. The W4A16 checkpoint is about 180 GB on disk and
   about 45 GB packed, so capture at TP>1, not TP=1: a
   `worker_extension_cls` (`ridgefill/capture_probe.py`) hooks the split
   layer inside every worker and accumulates the fit's sufficient
   statistics on rank 0. Capture at TP=4/EP=4, the `flashnext` serve shape;
   a bare TP=2/TP=4 load refuses the WNA16 MoE without expert parallelism.
   The map is replicated on every TP rank, so one fit serves every rank.
   `fit_stats.py` solves each layer's ridge fit from the statistics, picks
   lambda on the held-out windows (the paper's Flash-Next value is 0.03) and
   writes one bundle per split. A real capture refuses to run without a
   corpus: the 2026-11-03 capture omitted `--prompts-file`, so it fitted the
   script's placeholder prompts (one sentence repeated to 4096 tokens), not
   text. The refit on PG-19 books disjoint from the eval windows
   ([QUALITY.md](QUALITY.md), corpus) is pending its GPU slot.
3. **Path A.** Dequant int8 to fp16 at load (about 1.17 GiB per GPU if the
   map is replicated at TP=4) and apply with the existing fp16 GEMM, fp32
   accumulate. See [SILICON.md](SILICON.md). Path B, the HIP map kernel,
   is opt-in and off by default (`VLLM_RDNA_RIDGEFILL_MAP_FUSED=1`): 1.22x
   rocBLAS on fp16 dot2 and 2.51x on int8 dot4 (add
   `VLLM_RDNA_RIDGEFILL_MAP_MODE=dot4`) at 2048 rows. In the compiled TP4
   A/B (a413af6b52) the int8 map cuts 16K TTFT by 8.3 % against the rocBLAS
   fill at equal PPL. Every fill arm, though, is +0.72 long-window PPL
   against exact with maps fitted on placeholder text, so the quality
   verdict waits for a real-corpus fit ([TESTPLAN.md](TESTPLAN.md), item 4).
4. **The four HC streams are already full width on every TP rank.** No
   all-gather of `x_S`. Citation below.
5. **v1 guards.** `VLLM_RDNA_RIDGEFILL=1` refuses MTP and any other speculative
   config, and refuses prefix caching. Filled blocks are not stored in the
   CPU KV tier (`cpu_tier_may_store`). v2, off by default, salts filled
   blocks in `generate_block_hash_extra_keys` so a later fill-on request can
   reuse them and an exact request cannot.
6. **Hook.** `Qwen4ExpForCausalLM` on the ROCm class
   (`vllm/models/qwen4_exp/amd/model.py`). Not `qwen3_next.py`: that file is
   the 27B dense hybrid, which has no four-stream residual. The hook checks
   the width of the residual entering layer 24 and does not change it.
7. **27B AWQ and EXL3 second,** after a refit. The released 27B maps are
   llama.cpp only. Leave Mistral Large 4, DeepSeek-V4, SenseNova-Loop, and
   MiniMax. DeepSeek-V4.1-Flash is Leave for a different reason, under
   Related work: the split is trained in, and a ridge map adds nothing.

## Where the split state already lives

At TP=4 the recipe does not turn on HC sequence parallel.
`is_hc_sequence_parallel_enabled` is true only when TP > 1 and either
`enable_hc_sp` or sequence-parallel MoE is set
(`vllm/models/qwen4_exp/common/hyperconnection.py`, `is_hc_sequence_parallel_enabled`).
`enable_hc_sp` defaults false (`vllm/config/parallel.py`). Sequence-parallel
MoE also requires data parallel greater than 1. The flashnext recipe is TP=4
without that.

The residual entering a layer is `[tokens, hc_count * hidden]` with
`hc_count` 4 and `hidden` 2560, so 10240. The model expands the embedding
with `repeat(1, hc_count)` (`amd/model.py`, `Qwen4ExpModel.forward`). The HC
projections are replicated: `disable_tp=True` on the merged down/inject
linear, and `ReplicatedLinear` on the up projection
(`amd/hyperconnection.py`, `GatedResidual`). Attention `o_proj` and the GDN
`out_proj` are `RowParallelLinear` with `reduce_results=True` on this path,
and that all-reduce writes the full hidden width back on every rank
(`linear.py`, `RowParallelLinear.forward`). `rdna_ar` is that all-reduce
when the byte count is small; it still returns the full reduced tensor.

So `x_S` is the four streams entering layer 24, 10240 wide, on every rank:
the residual after layer 23's pending hyper-connection combine (which adds
its MoE output) and before layer 24's norm. The runtime and the capture
both materialize that combine at the split layer (ea1bb06a47). Before that
fix both read the streams without layer 23's MoE output, a state the exact
forward never holds. The paper does not say whether the fit saw the
pre-norm or post-norm vector. The V620 gate range-checks both the
10240-wide input and the 2560-wide output. See [TESTPLAN.md](TESTPLAN.md).

Sequence parallel, if someone turns it on later, shards the token axis, not
the hidden axis. That is a different design and not the flashnext recipe.

## Open choices

**(1) Refit, required.** Capture fp32 states in the dest image, fit offline,
write a new `ridgefill.json` whose pins are this checkpoint. The released
MXFP4 bundle will not load. `example_capture_fit.py` is the CPU synthetic
shape; `capture_activations.py` is the real tool. The fit uses
`math.fit_ridge`'s convention: center `x` and `y`, add
`alpha = lam * tr(G) / din`, solve `(Yc^T Xc) G^-1`, and set the bias. The
serve's `Session` applies with the fp16 GEMM. The server calls
`validate_pins` and raises on the first mismatch.

**(2) Indexer determinism.** Option A, take this: do not port
`patch-qsa-v3.diff` until two exact runs disagree. Option B: port the three
changes (canonical order, torch top-k above 2048 columns, NaN guard) and pay
the paper's 5–7% decode. The patch is Apache-2.0 inside ridgefill-data; it
is described in [ENGINE.md](ENGINE.md), not vendored.

**(3) MTP.** v1 forces it off. Filled rows have no final hidden state, and
the release ships no `final` map (`convert.py` says so). Option C2, later:
RidgeFill only during prefill, MTP only once decode is exact. Measure
acceptance only after the dtype pre-check in the test plan. Option C3, a
fitted `final` map, is not in this draft.

**(4) Prefix cache.** v1 forces it off while the flag is on. The flashnext
recipe actually turns prefix caching on (`FEATURES` includes
`prefix-caching`, and `ENABLE_PREFIX_CACHING` defaults to 1 in
`serve_rdna.sh`). The flag refuses that combination rather than trusting a
lock that is not in the launcher.

v2 (`VLLM_RDNA_RIDGEFILL_APC=v2`) salts a block that lies entirely in the filled
region. The cut is rounded down to the hash block, so the exact tail only
grows. The salt is `("ridgefill", split, map_sha, mode)` inside
`generate_block_hash_extra_keys` (`vllm/v1/core/kv_cache_utils.py`). An exact
request does not add it, so the hashes do not collide. A reused filled block
is the same approximation and may be reused only by a request that opted
into the same map. This is gated on the multi-turn trace. It is not the
default.

**(5) Chunks, graphs, TP, expert offload.** Chunks are 2048 tokens
(`MAXBAT=2048`). The tail is the last T tokens of the full prompt, not of
the chunk. A chunk that crosses the cut is mixed; v2 alignment avoids a
mixed hash block by making the cut a block boundary and running the rounded
tail exact. Graphs stay `FULL_AND_PIECEWISE`. The map GEMM is a large-M
prefill GEMM, so decode capture does not include it. First soak can use
`--enforce-eager` if a capture breaks. Expert offload on this fork is
optional (`rdna_expert_cache`, monokernel). The paper's speedup partly came
from reading routed experts over PCIe. On a V620 with experts resident, the
win is the skipped late MoE and attention, not that PCIe term.

**(6) Ranking.** Flash-Next, then 27B AWQ and EXL3 after a refit. Leave the
others, including V4.1-Flash (related work, not a port).

## Workloads

The target is a long single-shot prompt: RAG, document QA, a needle in a
long context. A prompt of 2048 tokens or fewer has an empty filled region
under tail 2048, so TTFT should not move.

Multi-turn chat and agent traffic whose prefix stays cached may not benefit.
The long prefill is rare if the cache is not invalidated. That point is
kishida's, and it does not depend on how DeepSeek V4.1-Flash is built. See
related work. The multi-turn row in the test plan compares fill on, fill
off, and fill off with prefix caching on. The last of those is a reference
number. It is not this fork's RidgeFill serve, and the stock flashnext
recipe's prefix caching is a separate question from the v1 guard.

## Clients

RidgeFill is server-only. Gateways and clients must not set
`VLLM_RDNA_RIDGEFILL` or any of its flags. Measure TTFT on the first streamed
chunk.

## Related work and community context

Posts in this section are community commentary. They are not measurements
and not paper results. Each one is linked. A full-archive search for the
string `RidgeFill` on 2026-11-02 returned no posts. The Zenodo preprint
record is dated the same day.

**DeepSeek-V4.1-Flash, trained split, Leave.** Fetched from
`https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/config.json`
(`model_type` `deepseek_v41`, 40 text layers). Verified fields:

- `kv_source_layer_ids` = `[2, 8, 14, 20]`
- `candidate_source_layer_id` = `20`
- `index_source_layer_ids` = `[2, 8, 14, 20, 24, 28, 32, 36]`
- `compress_ratios` has 43 entries. Indices 0–1 are 0, 2–19 are 2, 20–39
  are 1, and the last three are 0. `num_nextn_predict_layers` is 3, which
  matches the three extra zeros. `dspark_target_layer_ids` is `[37, 38, 39]`,
  inside the 40 layers, not those extra slots.
- `sliding_window` is 128.

Upper layers' KV sources stop at layer 20. That is a YOCO-style split
trained into the architecture: the upper layers were trained to read KV
from those source layers, and nothing is predicted. A ridge fill of late
inputs adds nothing, so V4.1-Flash is Leave for this PR. Serving the tail
replay in vLLM is a separate Later item and is not started here.

`https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/config.json`
(`model_type` `deepseek_v4`, 43 layers) has `compress_ratios` too, but the
pattern is 4 and 128, and it has none of `kv_source_layer_ids`,
`index_source_layer_ids`, or `candidate_source_layer_id`. The V4 paper
(arXiv 2606.19348) describes CSA, HCA, and SWA-128. It does not describe a
prefill layer skip. Do not read V4-Flash as RidgeFill.

kishida (`@kis`) posted on 2026-10-22 that V4.1-Flash's encoder-decoder mode
stops after the early layers and approximates the rest, with about 8B active
at prefill against 16B at decode, and that llama.cpp does not implement it.
Links:
[2113218159805436380](https://x.com/kis/status/2113218159805436380),
[2113216970321732082](https://x.com/kis/status/2113216970321732082),
[2113226575785365527](https://x.com/kis/status/2113226575785365527).
The config confirms a source-layer split at 20. It does not confirm
"approximates the rest" or an 8B active count. Those remain his commentary.
A further comment
([2098089379357179919](https://x.com/jatingargiitk/status/2098089379357179919))
describes about 8B at prefill plus a replay of the last 128 positions
through 20 decoder layers. The numbers 128 and 20 match `sliding_window` and
`candidate_source_layer_id`. The 8B figure and the replay procedure are not
in the config. Unverified commentary.

His workload point still stands, and it does not need the architecture to
be true: for a single user or a small group the KV cache is rarely
invalidated, long prefills are rare, and a prefill speedup that can
misbehave is a poor trade. That is why the workload section above prefers
long single-shot prompts over cache-friendly chat.

**Same stack, RDNA4.** Community posts show `tcclaviger/vllm` and radiance
on 2× or 4× R9700 with Qwen3.8 Flash-Next MXFP4-FP8-GPTQ and 27B MXFP4:
[benstadin](https://x.com/benstadin/status/2106551834068619745),
[this post](https://x.com/fel_rin/status/2111707164150829178),
[richgel999](https://x.com/richgel999/status/2108756251828183363),
[lonelyjew](https://x.com/lonelyjew/status/2106700520379629953).
R9700 is gfx1201 in this tree (`vllm/platforms/rocm.py`, device id
`0x7551`). gfx1030 has no FP8 hardware (`kHasFp8Wmma` is RDNA4-only in
`csrc/rocm/rdna/common/arch.cuh`; the
[W8A8 note](https://github.com/BlivionIaG/rdna-hip-wiki/blob/main/kernels/quant/w8a8.md)
says the same). That is why this draft ports the method and refits, and
does not run their image or their checkpoints.

**Closed-form KV maps across models. Background, not alternatives.**
NVIDIA's cross-model KV transfer (Heo et al., arXiv
[2608.03893](https://arxiv.org/abs/2608.03893), confirmed) fits a per-head
ridge map from one model's KV to a same-family sibling's. The paper's own
scope line: within-family transfer over dense full-attention models; hybrid
attention and attention-recurrent architectures are out of scope.
RaReCache (Sarkar et al., USC, UC Irvine, and Intel Labs, arXiv
[2610.11358](https://arxiv.org/abs/2610.11358), confirmed) runs prefill on a
small model, maps its KV, and recomputes some positions on the large model.
A community post describes a prefill-only layer skip with full-depth decode
([no_numb](https://x.com/no_numb/status/2099183057689362598)); it names no
paper. These produce attention KV. On Flash-Next, 18 of the 24 late layers
are GDN. A GDN layer keeps one recurrent state, not a per-token K and V, so
there is nothing for those maps to write. RidgeFill predicts the residual
and still runs the GDN update. That is why it is the method that fits this
hybrid, and why the KV-transfer papers stay background reading.

**YOCO-style serving ideas.** Community reports of a prefill speedup on
V4.1-Flash attributed to YOCO:
[MrCatid](https://x.com/MrCatid/status/2112693700560048562),
[qubitium](https://x.com/qubitium/status/2112939329467355224).
Take the idea that a trained early KV can serve the rest of prefill. Leave
their runtimes. vLLM's DSpark path
(`vllm/v1/worker/gpu/spec_decode/dspark/speculator.py`, and the V4.1 note in
`vllm/config/speculative.py`) is a draft stage bound to layers the
checkpoint names. It is part of the architecture, not a training-free fill.

The paper itself credits YOCO, DeepSeek-V4.1-Flash's causal encoder-decoder,
and kishida's Q3-8B-KVA-Projector as the idea's origin. Those citations are
in the Attribution section. The paragraph above is what this tree can check.

## Attribution and licenses

RidgeFill: Training-Free Late-Layer Cache Fill for Faster Prefill.
Dylan Johnston and Robert P. Smith.

- Paper: <https://openreview.net/forum?id=ymLpDkJIGs>
  (PDF <https://openreview.net/pdf?id=ymLpDkJIGs>).
  Zenodo preprint <https://doi.org/10.5281/zenodo.23890247>, resolved
  2026-11-02 to that title, Johnston and Smith, published 2026-11-02.
- Earlier technical note, v1.0 (2026-10-05):
  <https://doi.org/10.5281/zenodo.23179168>.
  The OpenReview PDF's only DOI is this one. The Flash-Next
  `ridgefill.json` `doi` field is this one. The map NOTICE also names
  23890247. This draft did not re-read every file in the data repo.
- Project page: <https://approximate-science.github.io/ridgefill/>
- Data: <https://github.com/Approximate-Science/ridgefill-data>
  tag `v1.8`, commit `f3b343fe267ce6376589b929615e34dfebf3f2fb`
  (annotated tag object `ecc6bf88`). Code Apache-2.0 (`LICENSE` and
  `code/LICENSE`). Results CC BY 4.0 (`LICENSE-DATA`). Both files were
  read at that tag.
- Maps: <https://huggingface.co/TheCacheIsALie/RidgeFill-Projector-Qwen3.8-Flash-Next-i8>
  and the 27B repo beside it. Apache-2.0 plus `LICENSE-Qwen` for the Qwen
  chat template. This draft does not copy the template or the tensors.
- llama.cpp reference:
  <https://github.com/Approximate-Science/llama.cpp-ridgefill>
  commit `b0439f7bb`. `LICENSE` at that commit is the MIT license of the
  ggml authors, copyright 2023-2026. No file from that tree is copied.
- Engine image used in the paper, patches not public, not inspected here:
  `tcclaviger/vllm:29.08.10` (digest in paper Table F.3). Describe it. Do
  not port it.
- radiance, <https://codeberg.org/StillDeadcode/radiance>, `LICENSE` on
  `main` is Apache-2.0. It is a separate engine. This draft does not use
  it and does not copy it.
- Idea the paper credits: kishida,
  <https://huggingface.co/kishida/Q3-8B-KVA-Projector>.

No source file in this change is adapted from those repositories. The fit
in `math.py` is an original implementation of the published normal
equation. If a later patch vendors `patch-qsa-v3.diff`, keep that file's
Apache-2.0 header and name `code/engine-fix/patch-qsa-v3.diff` at v1.8.

## What a GPU still has to do

Wire the prediction into the late layer: QSA writes K, V, and indexer keys
from `x_hat`; GDN runs projections and the state update from `x_hat` and
skips the MoE. Use the existing fp16 GEMM for the map. Run the test plan.
Do not land Path B, quality mode, or MTP-during-prefill on the back of this
draft.
