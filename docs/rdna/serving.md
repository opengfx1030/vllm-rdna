# Launch scripts and recipes

How to serve this fork. The entry point is [`tools/rdna/serve_rdna.sh`](../../tools/rdna/serve_rdna.sh). A recipe in [`tools/rdna/recipes/`](../../tools/rdna/recipes/) is the model, the parallelism, and the launch environment for one use case. Kernel-level defaults that a recipe does not touch are in [`docs/rdna2/tuning.md`](../rdna2/tuning.md).

```bash
MODEL=/path/to/checkpoint VENV=/path/to/venv \
  bash tools/rdna/serve_rdna.sh RECIPE=<name>
```

`PRINT=1` (or `--dry-run`) prints the environment, the TunableOp profile, and the exact command, then exits. Precedence, lowest to highest: recipe file, variables already in the environment, trailing `KEY=value` arguments.

```bash
PRINT=1 bash tools/rdna/serve_rdna.sh RECIPE=flashnext-mtp2 PORT=18100
```

Nothing in the launcher or the recipes bakes in a home directory, a host, or an IP. `MODEL` and `VENV` are required (`VENV` may be the active `VIRTUAL_ENV`).

## Launch environment, then clients

ROCm variables, `VLLM_*`, `HSA_*`, `NCCL_*` / `RCCL_*`, and vLLM serve flags belong to the process that runs `serve_rdna.sh`. An API client or a gateway (OpenAI SDK, OpenCode, LiteLLM, and others) receives the base URL and the served model name. It does not receive those variables or flags.

`GPU_MAX_HW_QUEUES` is one of those launch-environment variables. [`rdna_launcher_common.sh`](../../tools/rdna/rdna_launcher_common.sh) defaults it to **2**. Recipes that omit the key keep that default (`27b-awq`, `27b-exl3`, `dsv4-flash`, `full`, and both `flashnext-pp3*` recipes). The five Flash-Next TP=4 recipes — `flashnext`, `flashnext-exl3`, `flashnext-mtp0`, `flashnext-mtp2`, `flashnext-dflash2` — override it to **4**; leave that recipe value. Do not change those recipes to 2. A value of 2 as a host-specific fix is only an unverified workaround reported in [ROCm/TheRock#9306](https://github.com/ROCm/TheRock/issues/9306).

## Pick a recipe

| You want | Recipe | Parallelism | Listen |
| --- | --- | --- | --- |
| Flash-Next EXL3, tool and reasoning parsers | [`flashnext-exl3`](../../tools/rdna/recipes/flashnext-exl3.env) | TP=4 | `0.0.0.0:18094`, name `flash-next-exl3` |
| Flash-Next AWQ-W4A16, same parsers, no MTP | [`flashnext`](../../tools/rdna/recipes/flashnext.env) | TP=4 | `0.0.0.0:18094`, name `flash-next` |
| Flash-Next AWQ, plain decode, 262144 context | [`flashnext-mtp0`](../../tools/rdna/recipes/flashnext-mtp0.env) | TP=4 | `127.0.0.1:18096`, name `flash-next` |
| Flash-Next AWQ, MTP=2 | [`flashnext-mtp2`](../../tools/rdna/recipes/flashnext-mtp2.env) | TP=4 | `127.0.0.1:18096`, name `flash-next` |
| Flash-Next AWQ + Inco DFlash2 | [`flashnext-dflash2`](../../tools/rdna/recipes/flashnext-dflash2.env) | TP=4 | `127.0.0.1:18096`, name `flash-next` |
| Flash-Next AWQ on 3 cards, no MTP | [`flashnext-pp3`](../../tools/rdna/recipes/flashnext-pp3.env) | TP=1, PP=3 | `127.0.0.1:18096`, name `flash-next` |
| 3 cards, MTP=2 | [`flashnext-pp3-mtp2`](../../tools/rdna/recipes/flashnext-pp3-mtp2.env) | TP=1, PP=3 | `127.0.0.1:18097`, name `flash-next` |
| Qwen3.8-27B AWQ, W4A8 prefill | [`27b-awq`](../../tools/rdna/recipes/27b-awq.env) | TP=4 | `127.0.0.1:18210`, name `q27d` |
| Qwen3.8-27B EXL3, codebook `mul1` | [`27b-exl3`](../../tools/rdna/recipes/27b-exl3.env) | TP=4 | `127.0.0.1:18105`, name `exl3-27b-mul1` |
| DeepSeek-V4-Flash, K128 | [`dsv4-flash`](../../tools/rdna/recipes/dsv4-flash.env) | TP=4 | `127.0.0.1:18300`, name `dsv4-flash` |
| HIP graphs, breakable capture, TP=2 | [`full`](../../tools/rdna/recipes/full.env) | TP=2 | launcher omits `--host`; port `18094`; no served name |

The older `tools/rdna/serve_gfx1030_*.sh` files are aliases. They exec `serve_rdna.sh` with a fixed recipe and forward every argument:

| Script | Recipe |
| --- | --- |
| [`serve_gfx1030_flashnext.sh`](../../tools/rdna/serve_gfx1030_flashnext.sh) | `flashnext` |
| [`serve_gfx1030_flashnext_mtp.sh`](../../tools/rdna/serve_gfx1030_flashnext_mtp.sh) | `flashnext-mtp2` |
| [`serve_gfx1030_27b_dense.sh`](../../tools/rdna/serve_gfx1030_27b_dense.sh) | `27b-awq` |
| [`serve_gfx1030_exl3_27b.sh`](../../tools/rdna/serve_gfx1030_exl3_27b.sh) | `27b-exl3` |
| [`serve_gfx1030_full.sh`](../../tools/rdna/serve_gfx1030_full.sh) | `full` |

[`rdna_launcher_common.sh`](../../tools/rdna/rdna_launcher_common.sh) is the shared harness those entry points source (tree, venv, ROCm library path, TunableOp, attention, MTP capture sizes, stale-process teardown). Call `serve_rdna.sh`; do not assemble a `vllm serve` line by hand.

## What the launcher always does

From [`rdna_launcher_common.sh`](../../tools/rdna/rdna_launcher_common.sh), unless you already exported the variable:

| Variable | Value | When to change it |
| --- | --- | --- |
| `ROCM_SDK_ROOT` | `/opt/rocm/core-7.14` | Point it at the tree that contains the ROCm 7.14 SDK (`ROCM_HOME` and `HIP_PATH` follow it). |
| `HIP_VISIBLE_DEVICES` | `0,1,2,3` (`GPUIDS` is the other name) | The GPU list. Keep one tensor-parallel group on one PCIe switch. |
| `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE` | `1` | Set `HF_HUB_OFFLINE=0` only when this launch itself should reach the Hub. The usual path is to download first. |
| `VLLM_ROCM_USE_AITER`, `VLLM_ROCM_USE_AITER_MOE` | `0` | Leave off. AITER is a CDNA path. |
| `TORCH_BLAS_PREFER_HIPBLASLT` | `0` | Leave off. gfx1030 GEMMs go through rocBLAS + TunableOp. |
| `VLLM_WORKER_MULTIPROC_METHOD` | `spawn` | Leave it. |
| `ROCPROFILER_REGISTER_ENABLED` | `0` | Set `1` only while attaching a GPU profiler. |
| `VLLM_BATCH_INVARIANT` | `0` | Leave it for serving. |
| `VLLM_CACHE_ROOT` and the Triton / Inductor / extension caches | `<repo>/cache/...` | Stale `vllm serve` processes are killed only when they carry the same `VLLM_CACHE_ROOT`. |
| `PYTHONPATH` | the repo root | So this checkout wins over an installed copy. |

The command always includes `--dtype float16`. `TUNABLEOP` defaults to `1`: the launcher hashes `librocblas.so.5` and selects a profile under [`tunableop/`](../../tunableop/). `TUNABLEOP=0` turns that off. `TUNABLEOP_PROFILE=<name>` forces a named profile and refuses to start when the name is missing. `P2P=0` turns the RDNA all-reduce off and sets `NCCL_P2P_DISABLE=1` (RCCL over host shared memory) for a host whose PCIe peer-to-peer is broken.

`EXTRA_ARGS` is appended to the command. Example: `EXTRA_ARGS=--monokernel`. Quote it; the launcher splits it on spaces.

## Recipe keys

These are shell variables in the `.env` files. They become flags. They are not exported unless the name matches the launcher's environment prefixes (`VLLM_*`, `HSA_*`, and the rest listed in `serve_rdna.sh`).

| Key | Meaning | When to change it |
| --- | --- | --- |
| `SERVE_ENTRY` | `cli` (`python -m vllm.entrypoints.cli.main serve`) or `api` (the OpenAI API server module). | Either serves HTTP. The recipes already choose. |
| `SERVED_NAME` | `--served-model-name`. Empty means the client uses the checkpoint path. | Match whatever your client calls the model. |
| `HOST`, `PORT` | Bind address and port. An empty `HOST` omits `--host`. | `HOST=0.0.0.0` to leave localhost. |
| `TP`, `PP`, `PP_PARTITION` | Tensor parallel, pipeline parallel, and layers per stage (`VLLM_PP_LAYER_PARTITION`). | Three cards: a `flashnext-pp3*` recipe (`PP=3`, `TP=1`). Four cards: TP=4. |
| `MTP` | Speculative tokens. `0` is plain decode. The capture ladder becomes `(MTP+1) × {1,2,4,8}` unless `CG_SIZES` is set. | `2` on the MTP recipes. The tuning guide says to keep the ladder on powers of two. |
| `SPEC_CONFIG` | Replaces the MTP JSON and keeps that capture ladder. | `flashnext-dflash2` sets it. Override with a local drafter path. |
| `DFLASH_TOKENS` | Draft depth for `flashnext-dflash2` (default 3). Read from the caller environment while the recipe loads. | `3` favors 1k/512 at c=8; `2` favors 16k/1k at c=8 (recipe comment). |
| `ATTN` | `fa` → `RDNA_ATTN` and `VLLM_FA_RDNA2_GQA_DECODE=1`. `triton` → Triton AMD FA. `none` → the launcher does not pass `--attention-backend`. | Leave `fa` on the Flash-Next and 27B recipes. |
| `MAXLEN` | `--max-model-len`. Empty uses the model's own context. | Lower it when KV memory is tight. |
| `MIN_MAXLEN` | Refuse to start when `MAXLEN` is set below this. | `full` sets 32768. |
| `BLOCK_SIZE` | `--block-size`. | 1024 on the current Flash-Next and 27B recipes; 16 on `full`. |
| `SEQS` | `--max-num-seqs`. | Lower it when a long context does not fit. |
| `MAXBAT` | `--max-num-batched-tokens`. Empty: the launcher derives it from the KV and SSM dtypes (2048 for an FP16 KV cache, 4096 for `int4_fp16_block` with an FP16 SSM state, 6144 with the model's FP32 SSM state). A prefill chunk ends on a mamba block, so the budget covers two blocks. | Leave `27b-awq` empty. Pinning it puts a packed KV cache back on a one-block budget (see that recipe). |
| `LPTH` | `--long-prefill-token-threshold`. `0` disables the long-prefill split. | The recipes that set it use 0. |
| `PREFILL_INTERVAL` | `--prefill-schedule-interval`. | The tuning guide: admit prefills every Nth step so decodes get a pure-decode step. |
| `KV` / `KVCAP` | `--kv-cache-memory-bytes`. `KV=auto` drops the flag and needs `GMEM`, so vLLM sizes the cache after profiling. | Shrink it if the load fails to reserve memory. |
| `GMEM` | `--gpu-memory-utilization`. | `flashnext-pp3-mtp2` uses 0.975 so one 262144-token request fits a 30 GB card (recipe comment). |
| `DIST_TIMEOUT` | `--distributed-timeout-seconds`. | The recipes that set it use 1800. |
| `CG_MODE`, `COMPILE_MODE`, `CG_CAPTURE_SIZES`, `CG_EXTRA`, `CG_SIZES`, `COMPILATION_CONFIG` | CUDA-graph mode. `COMPILE_MODE=3` is inductor. `CG_CAPTURE_SIZES=1` lets the MTP helper fill the capture list. `COMPILATION_CONFIG`, when set, replaces that assembly. | Production is `FULL_AND_PIECEWISE`. `EAGER=1` passes `--enforce-eager` instead. |
| `EAGER` | `1` skips compile and graphs. | Correctness fallback. The tuning guide treats it as not a speed target. |
| `FEATURES` | Space-separated tokens, expanded below. | Add or remove a token on the command line. |
| `ENABLE_PREFIX_CACHING` | Used with the `prefix-caching` feature. `0` passes `--no-enable-prefix-caching`. | `full` sets 1. vLLM's own default is on; the token makes the flag explicit. |
| `W4A8` | Exported as `VLLM_RDNA2_W4A8_SDOT4`. | `27b-awq` sets 1 (int8-activation prefill, M ≥ 33; decode stays W4A16). `W4A8=0` on the command line falls back to W4A16 prefill. |
| `RDNA_AR` | Exported as `VLLM_RDNA_AR` when the recipe left `VLLM_RDNA_AR` unset. | `0` disables the one-shot all-reduce. `P2P=0` forces that too. |
| `PLE` | `bf16` (default), `int4`, `int4-disk`, `exl3`. | See [PLE](#ple). |
| `PLE_QUANT_DIR` | Directory of an INT4 PLE table (`META.json` required). | Required for `PLE=int4` and `PLE=int4-disk`. |
| `TUNED_CONFIG` | `1` sets `VLLM_TUNED_CONFIG_FOLDER` to `<repo>/tuned-moe`. | On for the AWQ MoE recipes. Off for `flashnext-exl3` (those configs are the W4A16 MoE path). |
| `ALLOC_CONF` | `expandable_segments` true, false, or `auto` (false when the PLE sidecar is on). | `full` uses `auto`. |
| `REQUIRED_VARS` | Names that must already be set or the launch stops. | No current recipe sets it. |

### `FEATURES` tokens

| Token | Flag |
| --- | --- |
| `trust-remote-code` | `--trust-remote-code` |
| `prefix-caching` | `--enable-prefix-caching` or `--no-enable-prefix-caching` |
| `mamba-align` | `--mamba-cache-mode align` |
| `prompt-tokens-details` | `--enable-prompt-tokens-details` |
| `auto-tool` | `--enable-auto-tool-choice --tool-call-parser qwen3_coder` |
| `reasoning` | `--reasoning-parser qwen3` |
| `expert-parallel` | `--enable-expert-parallel` |
| `language-model-only` | `--language-model-only` |
| `skip-mm-profiling` | `--skip-mm-profiling` |
| `vision-cap` | `--limit-mm-per-prompt` (default `{"image":1}`) and `--mm-processor-kwargs` (default `{"max_pixels":1605632}`) |

### PLE

Set on the launcher command, for Flash-Next AWQ. `flashnext-exl3` pins `PLE=exl3`.

| `PLE` | What the launcher exports | Table |
| --- | --- | --- |
| `bf16` | clears `VLLM_PLE_QUANT_DIR` and `VLLM_PLE_CPU_OFFLOAD` | Upstream pinned BF16 table. The launcher comment says ~95 GB host RAM. |
| `int4` | `VLLM_PLE_QUANT_DIR` | INT4 table from `PLE_QUANT_DIR` (`primitive-ai/Qwen3.8-Flash-Next-PLE-quant`, the `ples_int4` directory). Launcher comment: ~30 GB pinned. |
| `int4-disk` | `VLLM_PLE_QUANT_DIR` and `VLLM_PLE_CPU_OFFLOAD=1`, ready-timeout 3600 s | Same INT4 table, memory-mapped, CPU sidecar. |
| `exl3` | `VLLM_PLE_CPU_OFFLOAD=1`, ready-timeout 3600 s, no quant dir | `ngram_embedding.safetensors` beside the checkpoint. Recipe comment: 32.6 GB at ng5, page cache. |

Measured host-RAM peaks for the three AWQ modes are in the PLE section of [`docs/rdna2/tuning.md`](../rdna2/tuning.md). On the PP=3 recipes the PLE layers have to sit on stage 0; the BF16 table needs a large host, so those recipes' comments say to use `PLE=int4` or `int4-disk` on a smaller machine.

### KV cache dtype

Recipes leave the cache in fp16, except `dsv4-flash`, which passes `--kv-cache-dtype fp8`.

`--kv-cache-dtype int4_fp16_block` is implemented for `RDNA_ATTN` and is off unless you add it (`EXTRA_ARGS='--kv-cache-dtype int4_fp16_block'`). On `27b-awq`, leave `MAXBAT` unset so the launcher grows the prefill budget with the block. `--mamba-ssm-cache-dtype float16` in `EXTRA_ARGS` selects the 4096-token budget; without it the packed cache uses 6144. Format notes: [`docs/explore/kv-int4-fp16-scales/README.md`](../explore/kv-int4-fp16-scales/README.md). `int4_per_token_head` is a separate dtype in `vllm/config/cache.py`; the launcher's budget logic looks for the `int4_fp16_block` string.

### Decode-stall cap

`VLLM_RDNA_DECODE_STALL_MS` bounds how many prefill tokens share a step with running decodes. `0` disables it. It is a launch-environment variable.

| Recipe | Value in the file |
| --- | --- |
| `27b-awq` | `0` (the comment says the derived mamba budget already chunks prefill, and records the TTFT cost of turning the cap back on) |
| `27b-exl3` | `250` |
| `full` | unset; the comment says leave it off at TP=2, with `250` as the opt-in |
| Flash-Next AWQ and EXL3 recipes | commented out; the comment points at [`docs/rdna2/mixed-batch-stall-probe.md`](../rdna2/mixed-batch-stall-probe.md) for the opt-in of `250` |

QSA: `VLLM_RDNA_QSA_HIP` defaults to on (tuning guide). `flashnext-dflash2` sets it to `1` and also passes `--monokernel`. The MonoKernel's fused QSA store stands down while that variable is `1`; the three HIP `qsa_store_cache_rows` run instead ([tuning guide](../rdna2/tuning.md), `--monokernel` row).

## Environment the recipes share

Where a recipe sets these, the value is the one below. A later section names the exceptions.

| Variable | Value | Role |
| --- | --- | --- |
| `VLLM_USE_V2_MODEL_RUNNER` | `1` | V2 model runner. |
| `VLLM_RDNA_AR` | `1` | One-shot all-reduce up to 64 KiB, RCCL past that. |
| `VLLM_RDNA_AR_MAX_KB`, `VLLM_RDNA_AR_ONESHOT_KB` | `64` | Size gate. The recipe comments say two-shot is not boot-safe under PCIe load, so do not raise these. |
| `VLLM_RDNA_AR_BLOCKS`, `VLLM_RDNA_AR_PACE` | `0` | Set by the five Flash-Next TP=4 recipes, both PP=3 recipes, and `27b-awq`. |
| `VLLM_FORCE_CUSTOM_ALL_REDUCE` | `0` | Set by `27b-awq`, `27b-exl3`, `dsv4-flash`, `full`. The tuning guide says to keep the stock custom all-reduce off. |
| `HSA_FORCE_FINE_GRAIN_PCIE` | `1` | Fine-grain PCIe. Set by every recipe. |
| `HSA_ENABLE_SDMA` | `0` | Set by every Flash-Next recipe (TP=4 and PP=3) and by `27b-awq`. |
| `GPU_MAX_HW_QUEUES` | launcher default `2`; `4` in five recipes | `flashnext`, `flashnext-exl3`, `flashnext-mtp0`, `flashnext-mtp2`, and `flashnext-dflash2` set 4. Every other recipe omits it and keeps the launcher default. See [above](#launch-environment-then-clients). |
| `OMP_NUM_THREADS` | `4` | Set by every recipe except `27b-exl3` and `full`. |
| `TOKENIZERS_PARALLELISM` | `false` | Same recipes as `OMP_NUM_THREADS`. |
| `PYTHONFAULTHANDLER` | `1` | Same recipes as `OMP_NUM_THREADS`. |
| `VLLM_CAUSAL_CONV1D_RDNA2_FWD`, `VLLM_CAUSAL_CONV1D_RDNA2_UPDATE` | `0` | HIP causal conv1d off; Triton stays. Set by every Flash-Next recipe and by `27b-awq`. The tuning guide's code default is 1; these recipes override it. |
| `VLLM_ENABLE_STARTUP_PLAN` | `0` | Same recipes as the causal-conv pair. |
| `VLLM_RDNA_DENSE_INT8`, `VLLM_RDNA_DENSE_INT8_ONLY`, `VLLM_RDNA_DENSE_GEMV` | `0` on the TP=4 Flash-Next recipes and `27b-awq` | Both PP=3 recipes set `VLLM_RDNA_DENSE_INT8=1` and `VLLM_RDNA_DENSE_INT8_ONLY=1`, and leave `VLLM_RDNA_DENSE_GEMV=0`. |
| `VLLM_RDNA_FUSED_SE` | `1` | Fused shared expert, where the recipe sets it. `flashnext-exl3` sets `0` (EXL3 experts use `Exl3MoEMethod`). |
| `VLLM_GDN_HIP_PREFILL` | `1` on the Flash-Next AWQ recipes | `flashnext-exl3` sets `0`. |
| `VLLM_USE_AOT_COMPILE` | `0` and `VLLM_DISABLE_COMPILE_CACHE=1` | Per-rank JIT. Set by `flashnext-exl3`, `27b-awq`, `27b-exl3`, `dsv4-flash`, `full`. The recipe comments: a shared AOT cache replays one rank's device tensors onto the others. |

`VLLM_USE_BREAKABLE_CUDAGRAPH=1` is set only by `dsv4-flash` and `full`. The other recipes leave it unset. Copying it onto a Flash-Next recipe is how an old env dump fights the current graphs.

## Recipes

### `flashnext`

AWQ-W4A16 Flash-Next, TP=4, no MTP. This is the agent stack: `FEATURES` includes `auto-tool`, `reasoning`, `prompt-tokens-details`, and `vision-cap`, with 6 sequences, `KV=7000000000`, `GMEM=0.90`, `MAXBAT=2048`, `BLOCK_SIZE=1024`, `LPTH=0`, `DIST_TIMEOUT=1800`. `MAXLEN` is empty. `CG_MODE=FULL_AND_PIECEWISE`, `COMPILE_MODE=3`. `TUNED_CONFIG=1`. `PLE` stays at the launcher default (`bf16`) until you override it.

`SERVE_ENTRY=cli`, `HOST=0.0.0.0`, `PORT=18094`, `SERVED_NAME=flash-next`.

### `flashnext-exl3`

Same serving shape as `flashnext` (port, sequences, KV, features, graphs), for an EXL3 checkpoint. `PLE=exl3`. `MTP=0`. Expert parallel stays on: the recipe comment says EXL3 rotates 128-wide blocks of a 640-wide expert intermediate, so TP=4 cannot split an expert, and with expert parallel each rank holds whole experts.

Turned off relative to `flashnext`, because those paths are W4A16: `VLLM_RDNA_FUSED_SE=0`, `TUNED_CONFIG=0`, `VLLM_GDN_HIP_PREFILL=0`, `VLLM_ROCM_MOE_PREFILL=0`. AOT compile and the compile cache are off (`VLLM_USE_AOT_COMPILE=0`, `VLLM_DISABLE_COMPILE_CACHE=1`).

```bash
MODEL=/path/to/Qwen3.8-Flash-Next-exl3-3.05bpw \
VENV=/path/to/venv \
  bash tools/rdna/serve_rdna.sh RECIPE=flashnext-exl3
```

The checkpoint name in the recipe header is `turboderp/Qwen3.8-Flash-Next-exl3` (exllamav3 ≥ 1.4.5). Worked curl and the health check are in the [README quick start](../../README.md#quick-start).

### `flashnext-mtp0` and `flashnext-mtp2`

AWQ-W4A16 Flash-Next, TP=4, `SERVE_ENTRY=api`, `HOST=127.0.0.1`, `PORT=18096`, `SERVED_NAME=flash-next`. `MAXLEN=262144`, `SEQS=8`, `MAXBAT=2048`, `BLOCK_SIZE=1024`, `LPTH=0`, `PREFILL_INTERVAL=1`, `KV=4026531840`. No `GMEM` line, so the launcher does not pass `--gpu-memory-utilization`. Features: `expert-parallel prefix-caching mamba-align vision-cap`. Graphs match `flashnext`. `TUNED_CONFIG=1`. `PLE` defaults to `bf16`.

The only difference between the two files is `MTP=0` versus `MTP=2`. The recipe comments say MTP=0 wins at 8×1k/512 c=8 and MTP=2 wins at 8×16k/1k c=8. Those runs are the ones already written up in the tuning guide.

### `flashnext-dflash2`

AWQ-W4A16 target, external drafter `incoai/Qwen3.8-Flash-Next-DFlash2`. The drafter is not `MODEL`. Download it before serving (`HF_HUB_OFFLINE=1`). `DFLASH_TOKENS` defaults to 3 and is also written into `MTP` so the capture ladder matches the verification width. `SPEC_CONFIG` selects method `dflash`, that Hub id, and `"attention_backend":"RDNA_ATTN"`.

`EXTRA_ARGS=--monokernel`. `RDNA_DFLASH2_DRAFT_BF16=1` keeps the draft in BF16 (the tuning guide: the draft residuals overflow in FP16). `VLLM_RDNA_QSA_HIP=1`. Text only: `language-model-only skip-mm-profiling`, no vision cap. Otherwise the serving keys match `flashnext-mtp2` (port 18096, 262144, KV 4026531840, `TUNED_CONFIG=1`).

```bash
MODEL=/path/to/flash-next-awq VENV=/path/to/venv \
DFLASH_TOKENS=2 \
  bash tools/rdna/serve_rdna.sh RECIPE=flashnext-dflash2
```

Export `DFLASH_TOKENS` in the shell that runs the launcher, before the command. A trailing `DFLASH_TOKENS=2` is applied after the recipe file is sourced, so it does not rebuild `SPEC_CONFIG` or `MTP`.

### `flashnext-pp3`

AWQ-W4A16, three gfx1030 cards, `TP=1`, `PP=3`, `PP_PARTITION=16,16,16`, `MTP=0`. `SERVE_ENTRY=api`, port 18096, `MAXLEN=262144`, `SEQS=8`, `MAXBAT=2048`, `KV=auto`, `GMEM=0.95`. Text only (`language-model-only skip-mm-profiling`): the recipe comment says the vision encoder sits on stage 0 and does not fit beside 16 layers.

`VLLM_RDNA_DENSE_INT8=1` and `VLLM_RDNA_DENSE_INT8_ONLY=1` (the comment: keeping the fp16 copy next to the int8 shadow OOMs a stage). `VLLM_RDNA_DENSE_GEMV=0`. `TUNED_CONFIG=1`. `VLLM_GDN_HIP_PREFILL=1`, `VLLM_RDNA_FUSED_SE=1`. `HSA_ENABLE_SDMA=0` and both causal-conv variables are `0`, same as the TP=4 Flash-Next recipes. This file does not set `GPU_MAX_HW_QUEUES`.

```bash
bash tools/rdna/serve_rdna.sh RECIPE=flashnext-pp3 \
  MODEL=/path/to/awq PLE=int4 PLE_QUANT_DIR=/path/to/ples_int4
```

### `flashnext-pp3-mtp2`

Same three-card shape with `MTP=2`, `PP_PARTITION=17,17,14` (the drafter sits on the last stage), `PORT=18097`, `SEQS=4`, `MAXBAT=2112`, `KV=auto`, `GMEM=0.975`. `CG_SIZES='[3,6,12]'` replaces the default ladder. `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0`.

The recipe comment: 2112 is the 2048-token mamba block plus 4 sequences × 3 MTP tokens; at 2048 a 16K c=4 run admits only 3 requests. It needs the W4A16 drafter produced by [`tools/rdna/quant_mtp_experts.py`](../../tools/rdna/quant_mtp_experts.py). The comment says to use this recipe up to about four streams, and `flashnext-pp3` when you want more aggregate throughput. Dense int8 ONLY mode matches `flashnext-pp3`. `GMEM=0.975` is what the comment says fits one full 262144-token request on a 30 GB card; it says 32 GB cards fit the same request at 0.95.

### `27b-awq`

Qwen3.8-27B compressed-tensors AWQ-INT4, dense hybrid GDN, TP=4. `SERVE_ENTRY=cli`, `HOST=127.0.0.1`, `PORT=18210`, `SERVED_NAME=q27d`. `MTP=0`. `MAXLEN` empty. `BLOCK_SIZE=1024`. `SEQS=8`. `MAXBAT` empty on purpose. `KV=8000000000`, `GMEM=0.9`. `W4A8=1`, `RDNA_AR=1`. `EAGER=0`, `FULL_AND_PIECEWISE`. Features: `prefix-caching mamba-align language-model-only vision-cap`. `TUNED_CONFIG=1`. AOT compile off, for the same cross-rank reason as the EXL3 recipes.

`VLLM_RDNA_DECODE_STALL_MS=0`. The file's own comment is the place that records the 2026-10-25 W4A8 and stall measurements.

```bash
bash tools/rdna/serve_rdna.sh RECIPE=27b-awq \
  MODEL=/path/to/Qwen3.8-27B-AWQ W4A8=0
```

### `27b-exl3`

Qwen3.8-27B-exl3, 3.00 bpw, codebook `mul1`, TP=4. `SERVE_ENTRY=api`, port 18105, `SERVED_NAME=exl3-27b-mul1`. `MAXLEN=20480`, `MAXBAT=2048`, `SEQS=8`, `KVCAP=6000000000`, `GMEM=0.90`. `MTP=0`, `EAGER=0`, `ATTN=fa`, `FULL_AND_PIECEWISE`. Features: `language-model-only skip-mm-profiling trust-remote-code prefix-caching mamba-align`.

AOT compile off. `RDNA_AR=1` with the 64 KiB one-shot gate. RCCL for the larger messages: `NCCL_P2P_LEVEL=pxb`, `RCCL_P2P_NET_DISABLE=1`, `RCCL_P2P_BATCH_ENABLE=1`, `NCCL_PROTO=Simple`, `RCCL_MSCCL_ENABLE=0`, `VLLM_FORCE_CUSTOM_ALL_REDUCE=0`. `VLLM_RDNA_DECODE_STALL_MS=250`. `HSA_ENABLE_SDMA` is unset here. `EAGER=1` or `CG_MODE=PIECEWISE` are the overrides the recipe comment names.

### `dsv4-flash`

DeepSeek-V4-Flash at K128: fp8 dense weights and MXFP4 experts, fp16 attention through the RDNA DeepSeek-V4 path. `ATTN=none`, so the launcher does not force `RDNA_ATTN`. `EXTRA_ARGS="--kv-cache-dtype fp8"`. `MAXLEN=32768`, `SEQS=8`, `MAXBAT=2048`, `KV=6000000000`, `GMEM=0.90`, TP=4, port 18300, `HOST=127.0.0.1`, `SERVED_NAME=dsv4-flash`, `MTP=0`.

`COMPILATION_CONFIG` is a literal `FULL_AND_PIECEWISE` object with empty compile ranges (no `COMPILE_MODE`). `VLLM_USE_BREAKABLE_CUDAGRAPH=1`: the recipe comment says the model has no `torch.compile`'d submodule, and without the breakable graph the runner drops to `cudagraph_mode NONE`. `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`. AOT compile off. `TUNED_CONFIG=1`. The RCCL block matches `27b-exl3`. `VLLM_RDNA_AR=1` at 64 KiB.

The comment calls the checkpoint a 50%-pruned repack and says to judge it with [`tools/rdna/port_v031/ppl_probe.py`](../../tools/rdna/port_v031/ppl_probe.py) (perplexity about 8.4), not exact answers. Features: `language-model-only skip-mm-profiling trust-remote-code`. The comment's TP=8 example is an override, not the file default:

```bash
bash tools/rdna/serve_rdna.sh RECIPE=dsv4-flash TP=8 \
  MODEL=/path/to/dsv4-flash HIP_VISIBLE_DEVICES=2,3,4,5,6,7,8,9
```

### `full`

The recipe header calls this the full-HIP serve: breakable `FULL_AND_PIECEWISE` graphs, FA-RDNA2, W4A16, HIP KV, HIP GDN. It was checked on Qwen3.8-27B-AWQ-INT4 at TP=2 (comment: greedy PASS 3/3, 2026-09-09). `TP=2`. `HOST` empty. `PORT=18094`. `SERVED_NAME` empty. `MTP=0`. `ATTN=none`, while `VLLM_USE_RDNA2_FA=1` and `VLLM_FA_RDNA2_GQA_MODE=subgroup` are set in the environment (`subgroup` is the GQA prefill mode the comment says beat varlen/splitk). `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`. `VLLM_USE_BREAKABLE_CUDAGRAPH=1`.

`MAXLEN=200000`, `MIN_MAXLEN=32768`, `BLOCK_SIZE=16`, `SEQS=6`, `MAXBAT=2048`, `KV=7000000000`, `GMEM=0.90`, `DIST_TIMEOUT=1800`. `ALLOC_CONF=auto`. Features: `prefix-caching language-model-only skip-mm-profiling trust-remote-code`, with `ENABLE_PREFIX_CACHING=1`. AOT compile off. RCCL uses `NCCL_P2P_LEVEL=pix` (the TP=4 EXL3 and DeepSeek recipes use `pxb`). `VLLM_RDNA_AR=1` at 64 KiB, `VLLM_FORCE_CUSTOM_ALL_REDUCE=0`. The decode-stall cap is unset.

## Further reading

- [`docs/rdna2/tuning.md`](../rdna2/tuning.md) — code defaults, subsystem flags, TunableOp, and the measurements already recorded there.
- [`docs/rdna/monokernel/README.md`](monokernel/README.md) — MonoKernel scope on gfx1030.
- [`docs/rdna2/mixed-batch-stall-probe.md`](../rdna2/mixed-batch-stall-probe.md) — decode-stall cap.
- [gfx1030 wiki](https://blivioniag.github.io/gfx1030-wiki/) — host setup, the ROCm pin, and the published images.
