# Qwen3.8 Flash-Next from a GGUF checkpoint (UD-Q4_K_XL) on gfx1030

| Item | Value |
|---|---|
| Date | 2026-11-16 |
| Landed | #169 (loader, @crispyging) with #174 (plugin seam and workarounds), #177 (plugin fork marker); `rdna_extras` |
| Checkpoint | `unsloth/Qwen3.8-Flash-Next-GGUF`, folder `UD-Q4_K_XL/` (4 shards, 104 GiB) |
| Recipe | `flashnext-gguf-mtp0` (TP=4, MTP=0, text only) |
| Run on | 4x Radeon PRO V620, gfx1030, ROCm 7.14.0, FULL_AND_PIECEWISE |
| Not run | MTP (the file has no MTP head), vision, TP2/TP8/EP, gfx1100, CDNA |

## Serve

```bash
# Weights: the GGUF shards. MODEL: a directory with the model's HF config.json,
# tokenizer files and chat template (the loader reads no weights from it).
MODEL=/path/to/hf-config-dir \
GGUF_DIR=/path/to/UD-Q4_K_XL \
VENV=/path/to/venv \
  bash tools/rdna/serve_rdna.sh RECIPE=flashnext-gguf-mtp0
```

The recipe sets `--load-format qwen4exp_gguf --model-loader-extra-config {"gguf_dir": ...}` through `EXTRA_ARGS`, `MAXLEN=32768`, `KV=2147483648`. Weights take 22 GiB per card; 2 GiB of KV is 127,875 tokens. Raise `KV` or `MAXLEN` only if the card has the room.

## The plugin: use the fork

The loader runs the GGUF GEMMs through `vllm-gguf-plugin`, an out-of-tree package. Use **`opengfx1030/vllm-gguf-plugin`, branch `rdna_extras`**, built from source for gfx1030:

```bash
git clone -b rdna_extras https://github.com/opengfx1030/vllm-gguf-plugin
cd vllm-gguf-plugin
PYTORCH_ROCM_ARCH=gfx1030 pip install --no-build-isolation --no-deps .
pip install 'gguf>=0.17.0' 'huggingface_hub>=1.26.0'
```

The container image (`ghcr.io/opengfx1030/vllm-rdna`, `nightly` and `sha-*`) already has it, pinned in `docker/Dockerfile.rdna` (`GGUF_PLUGIN_REF`, `GGUF_PLUGIN_ARCH=gfx1030`).

The fork is upstream main plus:

- upstream PR #148 (`axiom-of-choice`), the Qwen3.5 config import that v0.0.5 needs on vLLM >= 0.32;
- the `moe_q` fix (below);
- `MOE_MMQ_MAX_EXPERTS` in `vllm_gguf_plugin/__init__.py`, which the loader's seam (`gguf_plugin_compat.py`) reads to pick the fast kernel. Order: the env var `VLLM_QWEN4EXP_GGUF_MMQ_MAX_EXPERTS`, then the plugin's marker, then 256.

With the PyPI plugin the loader is correct but slow (it slices big batches onto the per-slot kernel).

## What was wrong (three plugin kernel defects)

| # | Defect | Effect | Where handled |
|---|---|---|---|
| 1 | `topk_ids == -1` ("no expert" in vLLM) is read unchecked by `moe_vec_q` | GPU fault at the first CUDA-graph capture size, every worker | seam: routed to expert 0 with weight 0 |
| 2 | `moe_q` (MMQ, used for every batch above 64 tokens) returns early for expert ids above 255 (`moe.cuh`: `if (exp_idx > 255 \|\| exp_idx < 0) return;`) and its output starts at zero | The model has 512 experts: about half of the routed slots contributed nothing in all 48 layers. PPL 9.79 instead of 4.80, tokens flipped under batching | fork: bound removed. Seam default without the marker: slice batches to the per-slot kernel |
| 3 | The MMQ tile loader reads a few blocks past the end of the last row when K is not a multiple of its per-warp span (K 160 in the TP4 Q5_1/Q8_0 down projections) | The Q5_1 down tensor is 157,286,400 B = 75 x 2 MiB, so the read leaves the allocation: Memory Fault in `moe_q5_1` on every rank at boot | seam: 4 KiB of slack after the packed expert weights (only when MMQ is allowed) |

Kernel check on one V620 (real layer-0 experts, 512 experts, TP4 local shape, fp32 reference), cosine to reference:

| expert ids | per-slot kernel | MMQ, stock plugin | MMQ, fork |
|---|---|---|---|
| 0-255 | 0.9997 | 0.9997 | 0.9996 |
| 256-511 | 0.9996 | **0.0000** | 0.9996 |
| 0-511 | 0.9995-0.9997 | **0.68-0.76** | 0.9996 |

All 8,138 compared tensors (dense, experts of all 48 layers, PLE) agree with the AWQ checkpoint within quantization noise, so the loader's weight path was never the problem.

## Measured (4x V620, TP4, `flashnext-mtp0` settings, one boot each, KFD evictions +0)

| | stock plugin | seam slices (PyPI plugin) | fork plugin |
|---|---|---|---|
| greedy | pass | pass | pass |
| PPL (261 tokens) | 9.79 | 4.70 | **4.80** (AWQ 4.80-4.88) |
| concurrency probe | FAIL (2 of 48) | PASS | PASS (0 failing, 8 near-tie, 3 soft) |
| 1k/512 c=1 | TTFT 0.43 s, ITL 20.2 ms | TTFT 1.34 s, ITL 20.2 ms | TTFT 0.54 s, ITL 20.3 ms |
| 16k/512 c=1 | TTFT 6.58 s | TTFT 21.4 s | TTFT 8.16 s (2008 tok/s), ITL 20.6 ms |
| 1k/512 c=8 | not run | 143.3 tok/s | 186.85 tok/s, ITL 34.9 ms |

The first column is wrong output (see defect 2), shown only to explain the 9.79. The AWQ numbers on the same card are MTP2 (c=1 ITL 22.5-23.0 ms, TTFT 0.40 s at 1k and 5.93 s at 16k) and not matched to these. No AWQ MTP0 run with the same cells exists.

Per layer, MMQ against 64-token slices of the per-slot kernel: 1.39 vs 2.60 ms (128 tokens), 2.45 vs 10.39 ms (512), 7.95 vs 48.36 ms (2048).

The prefix-cache probe prints FAIL (2.0x: cold 1.00 s, warm 0.51 s) on every model: its 0.5 ratio gate does not fit a sub-second cold prefill (AWQ gives the same 1.9x). The cache works (hit rate 53 %).

## Other checks

- `#169` against `rdna_extras` on the non-GGUF path (AWQ `flashnext-mtp2`, TP4, `tools/rdna/pr_receipt.sh`, base `34e2ebc52a`, head `6c427d3e91`): token-exact 26/26, overlap and stall PASS, ITL p50 within +0.9 % / -1.4 % on all eight cells. The change adds 20 files and edits two (`model_loader/__init__.py`, `ngram_embedding.py`, gated on `gguf_ple_payload`); no `csrc`.
- Plugin dispatch, for reference: <= 64 tokens `ggml_moe_a8_vec` (per slot, no expert limit), above that `moe_align_block_size` + `ggml_moe_a8` (MMQ).
- Building notes: setuptools does not track `.cuh` header changes. After editing a plugin header, `rm -rf build` and delete the generated `*_hip.*` files before `pip install`.
- Pending upstream: the `moe.cuh` fix is a 2-line change; related plugin PRs are #130/#133 (negative expert ids), #146 (empty-block off-by-one) and #127. Nothing has been sent to `vllm-project/vllm-gguf-plugin`.
