# Inkling-2-Mini on the HI2 (rdna_hint_flex) expert format — runtime plan

Status: survey + design (2026-11-24). Branch `feat/inkling2-hi2-runtime`, based on
`rdna_extras` `1fdcb3a66e` (PR #195 is an ancestor; the Inkling-2-Mini port is already
merged there). Target: load and run the quantizer's production 3.5 bpw output
(`~/w4a8_runs/inkling-hi2/stage4/runs/bpw3.5`, 13.7 GiB) on 2 × V620 (gfx1030),
TP=2, language model first, vision and MTP after.

Everything below is read from source or headers unless marked **[measured]** (a
command was run) or **[inference]**.

## 1. Survey

### 1.1 What the fork has for Inkling-2 (non-HI2)

| Component | Where | What it does |
| --- | --- | --- |
| Model class | `vllm/models/inkling2_loop/amd/model.py:777` | `Inkling2LoopTextForCausalLM`; multimodal subclass in `amd/multimodal.py:9-145`. Requires fp16 (`model.py:806-816`). |
| Build-time expert format | `model.py:817` → `amd/loader.py:61` | Reads `hf_quant_config.json`; `expert_formats` (`loader.py:36-70`) accepts only `nvfp4` / `mxfp8` groups and raises otherwise. |
| Expert formats | `amd/moe.py:408` (`ExpertsNvfp4`), `:486` (`ExpertsMxfp8`) | Per-expert packed codes + scales repacked to kernel layout (`repack_nvfp4` `:345`, `repack_mxfp8` `:380`). |
| Expert kernels | `csrc/rocm/moe_lowp_dot2_rdna2.cu:437,453,742`; bindings `csrc/rocm/torch_bindings.cpp`; Python `vllm/_custom_ops.py` | `moe_nvfp4_gemm_rdna2`, `moe_nvfp4_gemm_swiglu_rdna2`, `moe_mxfp8_gemm_rdna2`; grouped 32-row prefill variant `moe_prefill_bm_rdna2.cu`. |
| Serving MoE | `moe.py:662` (`Inkling2LoopMoE`), `:759` repack, `:999` `prepare_hip`, `:1017` `forward_hip` | Decode-shaped grouped GEMM, block sizes by token count (`moe.py:606-620`), optional fused softcapped SwiGLU in the w2 load (`FUSE_GLUE`). |
| Activation | `moe.py:114` (`softcapped_swiglu`), `blocks.py:210-221`; softcaps `configs.py:59-60` | `softcap(gate,4) * sigmoid(gate) * softcap(up,25)` in fp32, cast back; sigmoid sees the raw gate. |
| Router / route_scale | `moe.py:75-101`, `configs.py:50` | sigmoid scores, top-k on `score + bias`, weights normalised, `route_scale` 8.0 applied. |
| Gate-first rows | `model.py` `copy_gate_up_deinterleaved` (`_GATE_UP_RE` path in `_load_tensor`) | Checkpoint `[g0,u0,...]` → `[gate; up]`; kernels' N axis is gate first. |
| Loop layers | `model.py:119`, `:507-537`, `:613-657`; `configs.py:114-140` | Layers 0-5 once, 6-17 × 3 passes, 18-23 once, final attention-residual readout. |
| MTP (DSpark) | `amd/dspark.py:3`, `:457`, `:563` | Draft reads `mtp.safetensors` through `allow_patterns_overrides = ["mtp.safetensors"]` (outside the index). |
| Vision | `amd/multimodal.py:41,145-160`, `amd/vision.py:295` | `model.visual.*` tensors streamed to the tower, bit-exact; the HI2 output carries them in `model-vision.safetensors` (indexed). |
| TP | `amd/parallel.py:181-220` (`check_supported`), `loader.py:112-135` (`local_slice`) | TP 1/2/4; experts split by intermediate units, attention by KV-head group; MoE returns a partial output that the caller all-reduces (`model.py:403`). |

### 1.2 What the fork has for HI2 MoE

* Generic method: `vllm/model_executor/layers/quantization/rdna_hint_flex.py` (config,
  dense linear) and `rdna_hint_flex_moe.py` (routed experts).
* Expert container: stacked per layer, `experts.gate_up.hi2_*` / `experts.down.hi2_*`
  with `hi2_part_off` and per-expert `had` / `col_scale`
  (`rdna_hint_flex_moe.py:55-82`, `absorb_stack` `:135-143`, `_stage_experts` `:144-241`).
* Field semantics and pure-torch dequant: `quantization/utils/rdna_hint_flex_utils.py`
  (fields table `:1-33`, `validate_stack` `:159`, `split_stack` `:181`,
  `select_cols` `:236` — 64-aligned, `dequantize_stored` `:316`, `act_transform` `:295`).
* Grouped prefill path: `_apply_grouped` (`rdna_hint_flex_moe.py:288-328`) uses
  per-pair transform + grouped GEMM building blocks; the decode/shared forward is the
  fused op `rdna_hint_flex_moe_rdna2` (`csrc/rocm/rdna_hint_flex_rdna2.cu:2537`).
* **The fused op hardcodes SiLU**: `at::silu(g13[:, :N]) * g13[:, N:]` at
  `rdna_hint_flex_rdna2.cu:2584` and `:2600`, so it cannot run Inkling's soft-capped
  two-argument activation. The **building blocks can**: `rdna_hint_flex_moe_transform_rdna2`
  (`:2682`) and `rdna_hint_flex_moe_gemm_rdna2` (`:2700`, `kMoeBlockM = 64` at `:2423`,
  `launch_moe_gemm` at `:2425`). Both launch on the current stream
  (`at::cuda::getCurrentCUDAStream`) with `C10_CUDA_KERNEL_LAUNCH_CHECK`, are compiled
  only when `VLLM_GPU_ARCHES` contains gfx1030 (`CMakeLists.txt:1781-1810`), and mixed
  INT2/INT4 row classes are already handled (the row headers carry `row_nbits` /
  `row_block`).
* No Inkling-specific HI2 path exists: `read_expert_formats` rejects anything but
  nvfp4/mxfp8, and no `hi2` mention under `vllm/models/inkling2_loop/`.

### 1.3 The quantizer's 3.5 bpw directory (measured on the box)

`config.json` (`quantization_config`):

```text
quant_method = rdna_hint_flex, version = 3.2, mode = flex,
target_bpw = 3.5, moe_layout = stacked,
moe_stacks = {gate_up: [gate_proj, up_proj], down: [down_proj]},
rotation = on, smooth = 0.5, rounding = gptq, calibrated = true
```

Tensor census (`model.safetensors.index.json`, shard headers only, **[measured]**):

| Class | Tensors | Bytes |
| --- | --- | --- |
| dense (text, HF names) | 645 | 2.994 GiB |
| experts (stacked HI2) | 480 = 24 layers × 2 groups × 10 arrays | 10.079 GiB |
| vision (`model-visual`... `model-vision.safetensors`) | 422 | 0.593 GiB |
| **total** | 1547 | **13.666 GiB** |

Per-layer expert stack shapes (layer 0, all layers analogous): `gate_up`
`codes I32 [71,020,368]`, `code_off I32 [294,913]`, `blk_params U8`, `param_off I32`,
`row_consts F16 [294,912,3]`, `row_nbits U8 [294,912]`, `row_block U8 [294,912]`,
`part_off I32 [257] (= E·2+1)`, `had U8 [128,144]` (K=2304), `col_scale F16 [128,2304]`;
`down` `had U8 [128,72]` (K=1152), `part_off I32 [129]`, `row_nbits [294,912]`.
`gate_up.row_nbits` over all 24 layers: INT2 2,982,071 rows, INT4 4,095,817 rows;
row blocks 16/32/64 all in use (layer 0: 247,780 / 43,159 / 3,973 rows). **[measured]**

Absent from the output: `hf_quant_config.json`; `mtp.safetensors` (the exporter
deliberately does not link an unindexed shard, `hi2/checkpoint.py:547-580`). Dense
text tensors are renamed to HF names (`model.language_model.*`,
`hi2/inkling_source.py:48-58`, `name_map` at `:172`); `lm_head.weight` keeps its name;
experts are `model.language_model.layers.N.mlp.experts.{gate_up,down}.hi2_*`.

### 1.4 Reconciliation with `InklingRuntimeScout`

The scout's report is correct on: no HI2 Inkling path, `read_expert_formats`
gate, HF vs serving names, no `hf_quant_config.json`, no MTP in the output, softcap /
route_scale / gate-first requirements, and the memory order of magnitude. Differences:

1. The scout could not find the PR #195 worktree. It is `vllm-inkling2-merge`
   (head `ee27e18374`); PR #195 is already an ancestor of `rdna_extras`
   `1fdcb3a66e`, so this branch starts from `rdna_extras`.
2. The scout estimated 3-5 agent-days for a softcap kernel. The existing grouped
   building blocks let the **first correct kernel path reuse the production HI2
   grouped GEMM** and apply the Inkling activation in torch between the two GEMMs
   (option b below). No new HIP code is required for correctness; a fused softcap
   kernel is an optimization, not a prerequisite. This removes the largest risk item.
3. The scout suggested reusing `rdna_hint_flex_moe_rdna2` as a whole. It cannot: the
   SiLU is hardcoded (`:2584`, `:2600`). The transform + grouped-GEMM pair is the
   reusable part.
4. The scout's TP concern (576 not divisible by 64) does not apply: TP=2 rank
   intermediate is 576, a multiple of 64 (and `hu.select_cols` requires 64-alignment,
   which 576 meets; 1152/2 = 576). TP=4 (288) would need kernel work and is out of
   scope.

## 2. Chosen design

**Loader-first; no quantizer change.** The 3.5 bpw directory is not modified. The fork
learns to read it:

1. **Format declaration (runtime side).** `amd/loader.read_quant_method(model, rev)`
   reads `config.json` and returns `quantization_config.quant_method`. When it is
   `rdna_hint_flex` (and `mode == "flex"`, `version == "3.2"`), `read_expert_formats`
   returns `["hi2"] * num_layers` after checking `moe_stacks` contains `gate_up` and
   `down`. The legacy `hf_quant_config.json` path is unchanged for NVFP4/MXFP8
   checkpoints. No quantizer-side change is needed; a `hf_quant_config.json` written
   by the exporter would be redundant (the vLLM config path is not used by the bespoke
   Inkling model). If the exporter later wants to serve generic vLLM tooling, writing
   one is a *separate, optional* small change request — not a blocker for this work.
2. **Name mapping (runtime side).** `loader.hf_to_serving()` inverts the reference
   `build_name_map` (`tools/rdna/inkling2/ref_checkpoint.py:166-260`) for the dense
   text tensors (`model.language_model.*` → `model.llm.*`) and maps the composed
   `model.language_model.message_input.query` directly to `model.message_input.query`.
   `lm_head.weight` and `model.visual.*` are unchanged. Expert groups are handled
   before the generic mapping.
3. **Expert container.** `amd/moe.py` gains a `quant_format == "hi2"` path:
   the loader collects the 10 arrays of each `gate_up`/`down` stack per layer
   (expert-group names as written by the quantizer), `process_weights_after_loading`
   canonicalises + validates them (`hu` semantics, reimplemented torch-only in
   `amd/hi2.py` so the module keeps its "torch only" import contract), slices them to
   this rank, and registers kernel-layout buffers:
   * `w13`: per expert, `gate rows [lo,hi)` then `up rows [I+lo, I+hi)`, kept in the
     kernel's `[E · 2N, ...]` order. K = hidden (not split by TP).
   * `w2`: all `H` rows per expert, `select_cols` K slice `[lo, hi)` (64-aligned).
   * `had [E,K/16]` and `col_scale [E,K]` follow the same slice.
   The full-stack CPU copy of one group is freed as soon as its 10 arrays have been
   sliced, so peak host memory is one layer's group (~0.5 GiB), not the whole model.
4. **Kernel path (option b).** `Inkling2LoopMoE.forward_hip` for `hi2`:
   router (existing fp32/fp16 router) → `moe_align_block_size(block 64)` →
   `rdna_hint_flex_moe_transform_rdna2(w13)` → `rdna_hint_flex_moe_gemm_rdna2(w13)` →
   `softcapped_swiglu` in fp32 (cast to fp16) → `transform(w2, top_k=1)` →
   `gemm(w2)` → weighted top-k sum in fp32. The grouped GEMM writes rows by routed
   pair id, so the top-k layout matches `ExpertsReference`.
   Rejected for now: (a) a fused soft-capped SwiGLU variant inside the grouped GEMM —
   it needs a new kernel and cannot be tested on a GPU this session; it stays as a
   follow-up optimization once the reference and the stock kernels agree. The eager
   `forward()` path dequantizes experts with the pure-torch `hu`/`hi2` decoder and is
   the correctness oracle (CPU tests); it is not used in serving.
5. **Gate for other platforms.** The kernel path is selected only when
   `current_platform.is_rocm()` and `on_gfx10x()` and the ops exist (same predicate as
   `rdna_hint_flex._rdna2_kernels_available`). Otherwise the loader still parses the
   directory (CPU tests) but `forward_hip` raises a clear error and the eager reference
   path is available. CUDA/CDNA/CPU code is untouched: no new `.cu`, no signature
   change, no change to NVFP4/MXFP8 classes.
6. **MTP.** `tools/inkling_hi2/prepare_serve_dir.sh` creates a **new** serve directory
   of symlinks: all HI2 files plus `mtp.safetensors` linked from the stock directory.
   It never writes into the quantizer output. `mtp.safetensors` stays out of the index
   (the DSpark loader reads it by pattern, `dspark.py:457`).
7. **Vision.** `model.visual.*` is in the index and bit-exact (measured by the
   quantizer); the existing tower loader consumes it unchanged. No repack.

### Memory fit at TP=2 (2 × V620, 29.98 GiB usable)

| Item | Per rank |
| --- | --- |
| HI2 experts 10.079 GiB / 2 (row slices of TP=2; K slice of w2) | ≈ 5.04 GiB |
| dense text (embed/unembed vocab-sharded, attention KV-head-sharded; router/norms replicated) | ≈ 1.6-2.2 GiB **[inference]** |
| vision tower, replicated when images are enabled | 0.59 GiB |
| serving workspace (1024 tokens) | 0.33 GiB (measured on the stock model, port spec) |
| full-attention KV, fp16 | 16 KiB/token/rank (8 full layers; 128 MiB at 8k, 512 MiB at 32k, 16 GiB at 1M) |
| sliding-window KV + conv state | 0.835 GiB fixed (window-bound, `1/tp` of 1.42 + 0.25 GiB) |
| **weights total** | **≈ 7.1-8.6 GiB** |

At 8k context a rank holds ≈ 9.6 GiB (with the conservative replicated-dense bound),
at 32k ≈ 10 GiB, leaving > 19 GiB for activations and the allocator reserve; even the
1M-token plan (≈ 25.4 GiB) fits the 29.98 GiB card only with the measured sharded
dense size and no extra transients **[inference]**. The load-time transient
(checkpoint format per-layer group before slicing, CPU) is bounded by one layer and
does not touch the device. The stock NVFP4 model measured 17.74 GiB model memory at
TP=1 (`docs/rdna2/inkling2-port-spec.md:240`); HI2 at TP=2 is smaller per rank.
`tools/inkling_hi2/memory_estimate.py` reproduces the table from the shard headers.

## 3. Work breakdown

| Step | Content | Files |
| --- | --- | --- |
| 1 | This survey/plan | `docs/rdna2/inkling-hi2-runtime-plan.md` |
| 2 | Format detection, HF→serving names, HI2 stack container/slicing, loader wiring, CPU tests (config declaration, name map, stack slice, memory census helper) | `vllm/models/inkling2_loop/amd/loader.py`, `amd/hi2.py` (new), `amd/model.py`, `amd/moe.py`, `tests/models/inkling2_loop/test_hi2_loader.py` (new) |
| 3 | Pure-torch HI2 expert reference (dequant + routed MoE: softcaps, route_scale, fp32 top-k sum) and a CPU test against the quantizer's dequant/eval on a tiny HI2 directory generated from the quantizer's tiny fixture | `amd/hi2.py`, `amd/moe.py`, `tests/models/inkling2_loop/test_hi2_reference.py` (new), `tools/inkling_hi2/make_tiny_hi2.py` (new, box-side fixture generator) |
| 4 | Kernel path: grouped transform + GEMM + softcapped activation + weighted sum in the eager forward; the same kernels from preallocated workspace buffers in `forward_hip`; GPU-marked unit test vs. the reference; exact GPU command | `amd/moe.py`, `amd/glue.py`, `amd/model.py`, `tests/models/inkling2_loop/test_hi2_kernel_gpu.py` (new) |
| 5 | GPU run kit: serve/smoke/KL/bench scripts, lease-quiet-boot protocol README, ordered checklist | `tools/inkling_hi2/{serve.sh,smoke.sh,kl_probe.py,bench.py,README.md}` (new) |

### Status (branch `feat/inkling2-hi2-runtime`)

* Steps 1-3 are committed and pushed: the plan, the loader (format declaration,
  name map, TP slices) and the float64/fp32 reference, proven against the
  quantizer's own tiny output: `test_hi2_loader.py` 23 passed,
  `test_hi2_reference.py` 4 passed (bit-identical decode, exact TP columns, routed
  MoE == transformers reference; 3m41s fixture).
* Step 4 is committed: `_forward_chunk_hi2` drives the generic
  `rdna_hint_flex_moe_transform_rdna2` + `..._gemm_rdna2` at block 64 with the
  Inkling soft-capped SwiGLU (`inkling2_glue_swiglu`) and a fp32 weighted sum;
  `_forward_hip_hi2` runs the same kernels from workspace views (`moe_x`, the w2
  output buffer, block-64 routing buffers). The CPU side is covered by the loader
  tests; the kernel test is `tests/models/inkling2_loop/test_hi2_kernel_gpu.py` and
  has **not run**: every GPU is leased.
* Step 5 is committed: `tools/inkling_hi2/{serve.sh,smoke.sh,kl_probe.py,bench.py}`
  and the ordered checklist in `tools/inkling_hi2/README.md`. Nothing in the kit
  has been executed on a GPU.
* Verifier round (`InklingRuntimeVerify`) fixed four start-up blockers on the real
  directory, each with a CPU regression test:
    * **TP=2 workspace**: the w2 output has its own `[T*K, H]` buffer
    (`moe_hidden_view`); the `hidden == 2 * intermediate` requirement is gone
    (real TP=2: hidden 2304, intermediate 576 per rank).
    * **block-64 routing**: `sorted_ids64` is rounded up to 64 (the HIP GEMM checks
    `numel % 64 == 0`); tested for every chunk size 1..1024 at TP=1 and TP=2.
    * **`lm_head.weight`**: the HI2 output keeps the stock top-level name; it is
    aliased to `model.llm.unembed.weight` in `load_weights`
    (`serving_checkpoint_name`).
    * **w2 top-k reduction** (introduced by the w2-buffer fix): `_sum_topk_rows`
    views the 2-D `[T*K, H]` w2 output as `[T, K, H]`, multiplies by the
    workspace's fp32 `[T, K]` `topk_w` and sums over K into `[T, H]`; a CPU test
    uses that exact view and a mutant is killed.
  Also: a runtime `on_gfx10x()` guard on both HI2 kernel paths (wider than the
  gfx1030-only kernels; documented residual); `validate_stack` refuses uneven
  expert rows for every part of every expert (gate/up at TP=1 included).
* Serving numerics: the w2 GEMM writes fp16, and the top-k reduction multiplies
  those fp16 products by the fp32 router weights in a `[tokens, top_k, hidden]`
  fp16 tensor before `torch.sum` (fp32 accumulator, fp16 output). The eager
  path/sum and the reference use fp32 products; the GPU kernel test and the KL probe
  compare them.

## 4. Risks / unknowns

1. **Grouped GEMM at decode sizes.** Block 64 per expert is correct but padded: an
   ~8-pair decode does 8 × 64 rows of work. Expect a slow first decode
   (**[inference]** several × to ~10× the stock kernel at c=1). Correctness first;
   the per-pair GEMV variant and a fused softcap kernel are follow-ups.
2. **CUDA-graph capture.** `forward_hip` writes preallocated workspace views (the
   block-64 routing layout and the w2 output included). `ops.moe_align_block_size`
   itself allocates one zero-element `expert_map` placeholder per call (as the
   generic MoE path does); the routing buffers it fills are preallocated. Capture is
   unproven on the device, so the kit boots with `FULL_AND_PIECEWISE` and falls back
   to `EAGER=1` only to report the difference. The `torch.mul_`/`torch.sum` epilogue
   is two captured torch kernels; if the graph rejects them the fallback is a small
   HIP epilogue.
3. **Load-time memory.** The loader reads the whole 13.7 GiB shard per rank (mmap);
   page cache is shared on the box. Host RSS peak is one stack group (~0.5 GiB) plus
   the dense tensors. Not measured on the real directory yet.
4. **Numerics.** The grouped kernel accumulates w13 in fp16 and writes the w2 output
   in fp16; the serving top-k reduction multiplies those fp16 products by the fp32
   router weights and sums with an fp32 accumulator, while the reference and the
   eager path multiply fp32 products. The kernel path's per-expert product is thus
   the fp16 GEMM output (inherent to the op). Tolerance and KL are proven only by
   the GPU probes; CPU tests compare the reference to the quantizer's decode.
5. **MTP.** The stock `mtp.safetensors` is bf16/stock-format; the HI2 text model keeps
   bf16 everything else, so the draft should load unchanged, but it has never run
   against HI2 experts. Language-model-only is the first GPU step.
6. **No GPU this session** (a2 leased by the quantization chain, b4 by other agents):
   kernel correctness, KL, memory and speed are unmeasured; the run kit exists so a
   leased session can execute the checklist without edits.

## 5. GPU session checklist (details in `tools/inkling_hi2/README.md`)

1. Lease slot A, quiet protocol, `boot.lock`; build my tree (MAX_JOBS ≤ 8, build.lock).
2. The HI2 output directory is directly servable (it carries the tokenizer);
   `validate` the directory and link the stock `mtp.safetensors` only for the later
   DSpark step.
3. CPU import/loader smoke on the box: tiny HI2 fixture round-trip.
4. Language-model-only, TP=2, `FULL_AND_PIECEWISE`, greedy smoke: load, one forward,
   16 decoded tokens; compare against stock-model greedy ids on the same prompts.
5. KL probe on the quantizer's held-out windows (reuse `hi2` ids / `inkling_kl_eval`
   windows); record KL, top-1 agreement, perplexity ratio.
6. Decode / prefill / memory cells, quiet protocol; record tok/s, TTFT, KV blocks,
   peak GPU memory, kernel counts.
7. Only then: vision + MTP smoke.
