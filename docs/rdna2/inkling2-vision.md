# Inkling-2-Mini vision on gfx1030 (V620)

Image input for `thinkingmachines/Inkling-2-Mini`
(`Inkling2LoopForConditionalGeneration`) in the RDNA2 fork: CPU preprocessing,
a vision tower that runs entirely in native HIP kernels
(`csrc/rocm/inkling2_vit_rdna2.cu`), and the vLLM multimodal glue.  Read
`inkling2-port-spec.md` first for the text stack.  Everything that is a guess is
marked **[INFERENCE]**; every number below was measured on the box (one V620,
HIP 2 of the `b4` lease) unless it says otherwise.

## What is delivered

| Area | File |
|---|---|
| kernels | `csrc/rocm/inkling2_vit_rdna2.cu` (+ `ops.h`, `torch_bindings.cpp`, `CMakeLists.txt` whitelist, `vllm/_custom_ops.py`: `inkling2_vit_*`) |
| encoder | `vllm/models/inkling2_loop/amd/vision.py` (`Inkling2VisionEncoder`) |
| preprocessing and vLLM processor | `vllm/models/inkling2_loop/amd/processing.py` |
| multimodal mixin (`embed_multimodal`, `embed_input_ids`, weight routing) | `vllm/models/inkling2_loop/amd/multimodal.py` |
| kernel tests | `tests/kernels/core/test_inkling2_vit_rdna2.py` |
| model tests | `tests/models/inkling2_loop/test_vision_{processing,loader,mixin,text_stack,encoder}.py` |
| oracle / bundle / GPU tool | `tools/rdna/inkling2/vision_check.py` (`hf-tiny`, `real`, `gpu`) |

`amd/model.py` and `registry.py` are not touched; the exact wiring patch is in
[Wiring](#wiring-for-the-lead).

## Request path

1. **Prompt.** The chat template renders an image as
   `<|content_image|><|unused_200054|>`; the processor keeps the marker (id
   200005) and expands the pad (id 200054) to one copy per 40x40 cell:
   `[200005] + [200054] * (rows * cols)`.  A pad directly behind a marker is
   dropped first, so a renderer that emits the marker alone and the HF template
   reach the same sequence.
2. **Pixels (CPU, `Inkling2ImageProcessor`).** No resize.  Each image is decoded
   (EXIF/mode handling of the vLLM media loader is bypassed when the encoded bytes
   are available), transparency is composited on the HF 8x8 checkerboard (204 /
   153), padded at the bottom/right to whole 40-pixel cells with raw `-1`, and cut
   into `[rows * cols, 40, 40, 3]` cells as fp16 (0..255 and -1 are exact).
   `vision_grid_thw = [1, rows, cols]`.  A 3082x2048 photo is grid `[1, 52, 78]`
   = 4056 decoder tokens (16 225 ViT tokens), 73 ms of CPU; an 800x600 RGBA PNG is
   `[1, 15, 20]`, 13 ms.
3. **Encoder (GPU).** `Inkling2VisionEncoder` runs the tower and projector; the
   output is fp16 `[rows * cols, 2304]` per image.
4. **Merge.** The mixin's `embed_input_ids` embeds the text (`embed_tokens` then
   `embed_norm`) and overwrites the pad rows with the image rows, so the vision
   rows bypass `embed_norm` as in the reference.  `test_vision_text_stack.py`
   checks, on the tiny checkpoint, that the text weights are bit-identical to the
   text-only class and that the loaded names are exactly the text names plus the
   tower's.

Only still images are supported (`grid_thw` t must be 1); audio and video are
rejected by the parser.

## Encoder structure

23 pre-LN blocks of width 1024 (16 heads of 64, MLP 4096, GELU, LayerScale) on
native 20x20 patches (four per 40x40 cell, 4 `rows * cols` patches plus one CLS
per image), interleaved axial 2-D RoPE, a learned 32x32 position grid
interpolated bilinearly to each image, `ln_pre`, a 2x2 patch merger and a
two-layer projector (4096 -> 2304).  Attention is bidirectional **per image**.

| Op | What it does |
|---|---|
| `inkling2_vit_gather_patches` | raw cells -> normalised fp16 patch rows (K padded 1200 -> 1216), two pixel modes |
| `inkling2_vit_gemm` | `C = A W^T`, 128x128 tile, `v_dot2_f32_f16`, fused epilogues: fp16 (+bias, +GELU), fp32 |
| `inkling2_vit_gemm_residual` | `res += gamma * (A W^T + bias)` on the fp32 residual |
| `inkling2_vit_embed` | patch GEMM + position embedding + CLS + `ln_pre` -> fp32 hidden |
| `inkling2_vit_layernorm` | fp32 rows -> LayerNorm (fp32 statistics) -> fp16 rows |
| `inkling2_vit_rope_tables`, `_rope_qk` | RoPE tables and in-place rotation of q and k |
| `inkling2_vit_attention` | per-image bidirectional flash attention, head_dim 64 |
| `inkling2_vit_pool` | non-CLS tokens -> fp16 rows for the merger |

The GEMM is the "D1" design of `docs/rdna2/hi2-prefill-gemm-2026-11-15.md`
(XOR-swizzled LDS, B-fragment dword rotation, buffer-descriptor loads) with the
epilogues fused.  Device code is compiled for gfx1030 only (empty kernels on the
other arches, so their code objects are unchanged); every launch is on the
current torch stream with `C10_CUDA_KERNEL_LAUNCH_CHECK()`.  There are no
atomics and the reduction orders are fixed, so every TP rank computes bitwise
identical embeddings for the same input (the batched result equals the
single-image result bitwise: `batch_equals_single_bitwise` is `True` for every
measured bundle).

* **Replicated per TP rank.** `supports_encoder_tp_data = False`; each rank holds
  the whole tower (318 256 128 parameters = 636 MB fp16) and runs the encoder
  redundantly.  Nothing is exchanged.
* **Eager.** The encoder is not part of the CUDA graph (variable shapes); the
  text stack keeps its graphs.
* **Chunked.** One launch encodes at most `MAX_CHUNK_TOKENS = 65 536` ViT tokens
  (whole images only); the workspace is about 22.25 KB per token (derived from
  the code), so at most about 1.46 GB, and every GEMM operand stays inside the
  kernels' 32-bit offsets.  An image above the cap is rejected.
* **Shapes.** Only head_dim 64, patch 20 and merge 2 are supported (guarded).

## Numerics and deviations

| # | Deviation from the bf16 reference | Effect |
|---|---|---|
| D1 | GEMM operands fp16, accumulation fp32, residual stream fp32, LayerNorm / softmax / GELU fp32, fp32 LN parameters, biases and LayerScale | the fp16 noise floor below |
| D2 | weights bf16 -> fp16 at load | exact for all but 60 487 of 318 256 128 values (0.019 %, magnitude below 6.1e-5, absolute error at most 3e-8) |
| D3 | q and k are rounded to fp16 before and after RoPE | part of the floor |
| D4 | attention probabilities are rounded to fp16 before `P V` (flash form) | not modelled by the fp16 emulation oracle; shows as the small gap native vs emulation |
| D5 | pooled merger input and the output are fp16 | the text stack consumes fp16 |
| D6 | pixel normalisation, see below | dominant deviation for smooth images |
| D7 | padding at the bottom/right with raw -1, as in the references | none |

### Pixel normalisation (D6): a deliberate default

The NVIDIA vLLM implementation (PR #69028, `_gather_patches_kernel`) does not
compute the HF processor's `2 v / 255 - 1`.  It computes, in IEEE fp32 without
FMA, `z = bf16((v / 255 - mean_c) / std_c)` and then
`out = bf16(2 * (z * std_c + mean_c) - 1)`: a CLIP normalisation rounded to
bf16, mapped back to [-1, 1] and rounded again.  The research notes warn against
silently replacing it, so the gather kernel has both and the encoder takes
`pixel_norm`:

* `"clip_bf16"` (**default**): the NVIDIA reference, bit-exact against a numpy
  fp32 implementation of the same formula for every raw value 0..255 and -1.
* `"exact"`: the transformers image processor, `2 v / 255 - 1`, used to validate
  the encoder against the HF fp32 model.

`vision_config["pixel_norm"]` selects the mode when the constructor argument is
absent; `--hf-overrides` replaces the whole `vision_config` dict, so the full
dict must be given.  Which of the two the *checkpoint* was trained with is
**[INFERENCE]** (the model card does not say); `clip_bf16` follows the only
serving implementation of the model.

Measured effect (fp32 oracle, the same weights, the two pixel paths against each
other, 11 images of the validation set: this is the size of the choice, not an
error of the kernel):

| image | rows | cos min | mean abs | max abs / rms | rel. row mean |
|---|---|---|---|---|---|
| hato 640 px crop | 256 | 0.99862 | 9.13e-4 | 0.307 | 9.28e-3 |
| hato + RGBA, 640 px | 496 | 0.99597 | 8.44e-4 | 0.312 | 8.52e-3 |
| hato 1600 px crop | 1600 | 0.93527 | 1.49e-3 | 1.297 | 1.58e-2 |
| white | 256 | 0.99266 | 2.55e-3 | 0.642 | 2.98e-2 |
| black | 256 | 0.99809 | 1.40e-3 | 0.287 | 1.59e-2 |
| noise | 256 | 1.00000 | 1.21e-3 | 0.015 | 1.87e-3 |
| checker | 256 | 1.00000 | 4.79e-3 | 0.006 | 1.26e-3 |
| text | 240 | 0.99953 | 2.29e-4 | 0.110 | 2.46e-3 |
| gradient | 270 | 0.89936 | 8.83e-3 | 1.698 | 8.89e-2 |
| odd size | 66 | 0.99534 | 1.74e-3 | 0.550 | 1.76e-2 |
| mixed (odd + text + gradient) | 576 | 0.89936 | 4.43e-3 | 1.762 | 4.47e-2 |

Smooth images (gradient, white, large crops) are the sensitive ones: a 1-LSB bf16
difference of the input moves whole rows.  For scale, the fp16 arithmetic floor
of the same images in `clip_bf16` mode is 0.99987 cos min / 1.3e-4 mean abs for
the hato crop and 0.99996 / 1.7e-4 for the gradient, and the transformers model
in bf16 sits at 0.9949 / 2.2e-3 (hato crop) and 0.9525 / 5.5e-3 (gradient) from
its own fp32: the pixel path is the dominant numeric deviation after the
reference's own bf16 noise.

In `exact` mode the floor is lower (0.99965 / 1.7e-4 for the hato crop, 0.9975
for the 1600 px crop, 0.99941 / 7.6e-4 for the gradient) because rounding the
normalised pixel to fp16 is then part of the arithmetic; in `clip_bf16` mode the
pixels are bf16 values, which fp16 holds exactly.

## Accuracy

Oracles (offline, CPU, `tools/rdna/inkling2/vision_check.py`; the only PyTorch
compute in this work):

* `fp32 oracle`: the tower in fp32 with the same pixel mode.  It matches the
  transformers `Inkling2` vision model in fp32 to `max_abs <= 2.7e-6`
  (`hf-tiny`, both pixel modes, cos min 1 - 2e-12) and to `<= 2e-5` on the real
  checkpoint and real images.
* `emulation oracle`: the same arithmetic as the kernels (fp16 operands, fp32
  accumulation, fp16 roundings at the same points, D4 excepted).  Its distance
  to the fp32 oracle is the **fp16 floor**.
* HF bf16 (the transformers model in bf16 on CPU) is the noise of the reference
  precision, for scale.

Gate (`vision_check.within_floor`, native vs fp32 oracle, relative to the floor of
the same image): `cos_min >= max(0.997, floor_cos_min - 2e-3)`,
`rel_row_mean <= 2 x floor`, `max_abs / rms <= 2 x floor + 0.02`.  The
random-weight embed-contract test (`test_vision_encoder.py`) also keeps absolute
caps (`rel_row_mean <= 4e-3`, `max_abs / rms <= 0.25`).  The criteria are
floor-relative because the minimum cosine falls with the number of rows and with
smooth images: the floor of the gradient image already exceeds the absolute
`rel_row_mean` cap, which is why the first, absolute gate failed it.

Reference noise (CPU oracles, `clip_bf16` pixels; `emu` is the fp16-emulating
oracle against fp32 = the fp16 floor; `HF bf16` is the transformers model in bf16
against its own fp32 on the same image, measured where run; the last column is
the fp32 oracle against the transformers fp32 model):

| image | emu cos min | emu mean abs | emu max abs / rms | emu rel. row mean | HF bf16 cos min | HF bf16 mean abs | HF bf16 max abs / rms | oracle vs HF fp32 max abs |
|---|---|---|---|---|---|---|---|---|
| hato 640 px crop | 0.99987 | 1.32e-4 | 0.065 | 1.34e-3 | 0.9949 | 2.24e-3 | 0.417 | 5.6e-6 |
| hato + RGBA, 640 px | 0.99987 | 1.34e-4 | 0.064 | 1.36e-3 | | | | 6.2e-6 |
| hato 1600 px crop | 0.99986 | 1.36e-4 | 0.057 | 1.41e-3 | 0.9753 | 2.72e-3 | 0.580 | 3.4e-5 |
| white | 0.99998 | 1.66e-4 | 0.023 | 2.12e-3 | | | | 3.5e-6 |
| black | 0.99989 | 2.80e-4 | 0.079 | 3.35e-3 | | | | 8.6e-6 |
| noise | 1.00000 | 1.50e-4 | 0.001 | 2.32e-4 | | | | 1.7e-6 |
| checker | 1.00000 | 5.53e-4 | 0.001 | 1.45e-4 | | | | 7.6e-6 |
| text | 0.99999 | 9.60e-5 | 0.012 | 1.02e-3 | 0.9967 | 1.39e-3 | 0.318 | 3.6e-6 |
| gradient | 0.99996 | 1.69e-4 | 0.034 | 1.70e-3 | 0.9525 | 5.45e-3 | 1.518 | 9.1e-6 |
| odd size | 1.00000 | 1.04e-4 | 0.009 | 1.07e-3 | | | | 2.2e-6 |
| mixed | 0.99996 | 1.31e-4 | 0.035 | 1.35e-3 | | | | 1.1e-5 |

Native kernels vs the oracles, production `_rocm_C`, one V620.  Each cell is
`native vs fp32 oracle / floor` (floor = fp16-emulating oracle vs fp32 of the
same image); the last column is native vs the fp16-emulating oracle (the
distance that D4 and the summation order leave).  `clip_bf16` (default):

| image | rows | cos min | mean abs | max abs / rms | rel. row mean | cos min vs emu |
|---|---|---|---|---|---|---|
| hato 640 px crop | 256 | 0.99984 / 0.99987 | 1.52e-4 / 1.32e-4 | 0.071 / 0.065 | 1.55e-3 / 1.34e-3 | 0.99995 |
| hato + RGBA, 640 px | 496 | 0.99984 / 0.99987 | 1.56e-4 / 1.34e-4 | 0.070 / 0.064 | 1.57e-3 / 1.36e-3 | 0.99985 |
| hato 1600 px crop | 1600 | 0.99992 / 0.99986 | 1.63e-4 / 1.36e-4 | 0.041 / 0.057 | 1.70e-3 / 1.41e-3 | 0.99967 |
| white | 256 | 0.99999 / 0.99998 | 1.70e-4 / 1.66e-4 | 0.018 / 0.023 | 2.17e-3 / 2.12e-3 | 0.99999 |
| black | 256 | 0.99994 / 0.99989 | 2.80e-4 / 2.80e-4 | 0.060 / 0.079 | 3.35e-3 / 3.35e-3 | 0.99996 |
| noise | 256 | 1.00000 / 1.00000 | 2.00e-4 / 1.50e-4 | 0.003 / 0.001 | 3.15e-4 / 2.32e-4 | 1.00000 |
| checker | 256 | 1.00000 / 1.00000 | 9.02e-4 / 5.53e-4 | 0.003 / 0.001 | 2.52e-4 / 1.45e-4 | 1.00000 |
| text | 240 | 0.99996 / 0.99999 | 1.37e-4 / 9.60e-5 | 0.031 / 0.012 | 1.45e-3 / 1.02e-3 | 0.99993 |
| gradient | 270 | 0.99996 / 0.99996 | 1.89e-4 / 1.69e-4 | 0.034 / 0.034 | 1.95e-3 / 1.70e-3 | 0.99999 |
| odd size | 66 | 1.00000 / 1.00000 | 1.21e-4 / 1.04e-4 | 0.016 / 0.009 | 1.23e-3 / 1.07e-3 | 1.00000 |
| mixed (3 images) | 576 | 0.99996 / 0.99996 | 1.59e-4 / 1.31e-4 | 0.035 / 0.035 | 1.66e-3 / 1.35e-3 | 0.99993 |
| hato 3082x2048 (full) | 4056 | 0.99988 / 0.99984 | 1.74e-4 / 1.44e-4 | 0.080 / 0.098 | 1.75e-3 / 1.45e-3 | 0.99981 |

`exact` pixels (the transformers processor path; includes the full-resolution
3082x2048 photo, 4056 rows):

| image | rows | cos min | mean abs | max abs / rms | rel. row mean | cos min vs emu |
|---|---|---|---|---|---|---|
| hato 640 px crop | 256 | 0.99983 / 0.99965 | 1.86e-4 / 1.70e-4 | 0.089 / 0.130 | 1.88e-3 / 1.72e-3 | 0.99996 |
| hato + RGBA, 640 px | 496 | 0.99983 / 0.99965 | 1.80e-4 / 1.60e-4 | 0.087 / 0.127 | 1.81e-3 / 1.61e-3 | 0.99996 |
| hato 1600 px crop | 1600 | 0.99835 / 0.99750 | 2.35e-4 / 2.16e-4 | 0.164 / 0.203 | 2.45e-3 / 2.27e-3 | 0.99960 |
| hato 3082x2048 (full) | 4056 | 0.99913 / 0.99895 | 2.51e-4 / 2.30e-4 | 0.142 / 0.168 | 2.55e-3 / 2.35e-3 | 0.99969 |
| white | 256 | 0.99999 / 0.99998 | 1.68e-4 / 1.65e-4 | 0.020 / 0.024 | 2.14e-3 / 2.13e-3 | 0.99999 |
| black | 256 | 0.99993 / 0.99996 | 2.96e-4 / 2.67e-4 | 0.072 / 0.035 | 3.57e-3 / 3.21e-3 | 0.99996 |
| noise | 256 | 1.00000 / 1.00000 | 2.22e-4 / 1.77e-4 | 0.003 / 0.001 | 3.49e-4 / 2.74e-4 | 1.00000 |
| checker | 256 | 1.00000 / 1.00000 | 9.02e-4 / 5.53e-4 | 0.003 / 0.001 | 2.52e-4 / 1.45e-4 | 1.00000 |
| text | 240 | 0.99996 / 0.99998 | 1.29e-4 / 1.01e-4 | 0.030 / 0.022 | 1.37e-3 / 1.09e-3 | 0.99989 |
| gradient | 270 | 0.99942 / 0.99941 | 7.74e-4 / 7.60e-4 | 0.175 / 0.184 | 7.64e-3 / 7.48e-3 | 0.99997 |
| odd size | 66 | 0.99999 / 0.99998 | 1.82e-4 / 1.73e-4 | 0.021 / 0.021 | 1.86e-3 / 1.77e-3 | 1.00000 |
| mixed (3 images) | 576 | 0.99942 / 0.99941 | 4.38e-4 / 4.19e-4 | 0.181 / 0.191 | 4.37e-3 / 4.16e-3 | 0.99989 |

All 24 cases pass `within_floor`; the native kernels stay inside the fp16 floor
(`cos min` never more than 3e-5 below it, `rel. row mean` at most 1.74x it, on the
checker image whose floor is 1.5e-4; the gate allows 2x).
`batch_equals_single_bitwise` is `True` in every case (a batched call returns
bit-identical rows to the single-image calls).

### Per-layer activation ranges (real checkpoint, real and synthetic images)

fp16 overflows at 65 504.  Maximum absolute value of every stage, per image
(`clip_bf16` pixels; the overall maxima in `exact` mode differ by at most 0.1 in
every stage):

| stage | embed `ln_pre` in | q | k | v | attn out | attn proj | resid after attn | fc1 | gelu | fc2 | resid after mlp | proj in | proj out |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hato 640 px crop | 14.8 | 12.5 | 17.7 | 7.1 | 6.2 | 15.4 | 48.8 | 50.6 | 37.1 | 32.8 | 48.9 | 4.0 | 2.2 |
| hato 1600 px crop | 14.8 | 14.6 | 16.7 | 7.4 | 7.2 | 15.9 | 48.3 | 56.0 | 47.4 | 33.6 | 48.4 | 3.3 | 2.4 |
| noise | 28.2 | 12.4 | 17.7 | 7.7 | 6.9 | 10.1 | 46.6 | 45.9 | 34.3 | 32.1 | 46.7 | 4.6 | 6.5 |
| checker | 57.4 | 10.8 | 16.8 | 6.5 | 6.5 | 11.7 | 66.5 | 27.1 | 20.7 | 30.4 | 66.5 | 23.3 | 33.1 |
| **max over 11 images** | 57.4 | 14.6 | 17.7 | 8.6 | 8.5 | 15.9 | 66.5 | 63.3 | 47.4 | 33.6 | 66.5 | 23.3 | 33.1 |

The smallest fp16 headroom over all images and stages is 985x (resid after mlp of
the checker image).  The residual stream is fp32 anyway; no fp16 operand exceeds
66.5.

## Performance (one V620, production `_rocm_C`)

Timings are the mean of three back-to-back encodes after one warm-up call (host
planning and all launches, synchronised at the end, weights resident).  The pixel
mode does not touch the GEMM or
attention kernels.  Peak of `v_dot2_f32_f16` on 72 CUs is 18.43 TF/s per GHz,
30.0 TF/s at 1.63 GHz **[INFERENCE: the clock was not logged]**, so the 17 to
21 TF/s below are 57 to 71 % of that **[INFERENCE]**.

| input | decoder tokens | ViT tokens | time | achieved |
|---|---|---|---|---|
| odd size | 66 | 265 | 16.4 ms | 10.0 TF/s |
| text image | 240 | 961 | 39.5 ms | 16.7 TF/s |
| hato 640 px crop | 256 | 1025 | 41.2 ms (38.7 ms in the `exact` run) | 17.2 TF/s |
| hato + RGBA, 640 px (2 images) | 496 | 1986 | 70.0 ms | 19.5 TF/s |
| 3-image batch | 576 | 2307 | 78.3 ms | 20.1 TF/s |
| hato 1600 px crop | 1600 | 6401 | 358.5 ms | 21.4 TF/s |
| hato 3082x2048, full resolution (`exact` run) | 4056 | 16 225 | 1665 ms | 20.7 TF/s |
| hato 3082x2048, full resolution (`clip_bf16` run) | 4056 | 16 225 | 1657 ms | 20.8 TF/s |

Microbench of the kernels (M x N x K of the encoder's GEMMs, ms and TF/s; the
torch column is an offline reference only and is not used in serving):

| shape (M = 1025) | native | torch fp16 linear |
|---|---|---|
| patch 1216 -> 1024 | 0.110 ms, 23.2 | 0.118 ms, 21.6 |
| qkv 1024 -> 3072 | 0.298 ms, 21.7 | 0.253 ms, 25.5 |
| out 1024 -> 1024 | 0.126 ms, 17.1 | 0.089 ms, 24.2 |
| fc1 1024 -> 4096 | 0.352 ms, 24.4 | 0.334 ms, 25.7 |
| fc2 4096 -> 1024 | 0.381 ms, 22.6 | 0.331 ms, 26.0 |

At M = 4801 the native GEMMs reach 33.3 / 28.6 / 19.8 / 24.8 / 24.3 TF/s.  The
attention kernel: 14.7 TF/s at L = 1025, 23.9 at L = 4801, 19.5 at L = 10 817
(torch SDPA in fp16: 2.3 and 1.5 TF/s at the first two).  The small `out`
projection (N = K = 1024, fp32 residual epilogue) is the least efficient GEMM.

## Tests

CPU (box CPU, `HIP_VISIBLE_DEVICES=`):

```bash
PYTHONPATH=$PWD:<transformers main> python -m pytest -q -p no:cacheprovider \
  tests/models/inkling2_loop/test_vision_processing.py \
  tests/models/inkling2_loop/test_vision_loader.py \
  tests/models/inkling2_loop/test_vision_mixin.py \
  tests/models/inkling2_loop/test_vision_text_stack.py
python tools/rdna/inkling2/vision_check.py hf-tiny
```

66 passed (32 + 20 + 11 + 3) in 40.2 s; `hf-tiny` prints PASS for both pixel modes
(oracle vs transformers fp32 `max_abs` 2.6e-6 / 2.4e-6, cos min 1 - 2e-12).

GPU (b4 lease, `gfx1030`):

```bash
python -m pytest -q tests/kernels/core/test_inkling2_vit_rdna2.py
python -m pytest -q tests/models/inkling2_loop/test_vision_encoder.py
python tools/rdna/inkling2/vision_check.py gpu --real --reps 3 --ref <bundle.pt>...
```

Results with the final sources (production `_rocm_C` built in the tree, one V620
on HIP 2 of the `b4` lease):

| run | result |
|---|---|
| `import vllm._rocm_C` | all nine `inkling2_vit_*` ops registered; `AMD Radeon Pro V620`, torch 2.12.0+rocm7.14.0 |
| kernel tests | 69 passed in 23.9 s |
| encoder tests (`test_vision_encoder.py`) | 11 passed in 26.9 s |
| oracle bundles, `clip_bf16` | 11 of 11 PASS |
| oracle bundles, `exact` | 12 of 12 PASS |
| oracle bundle, `clip_bf16`, full 3082x2048 photo | 1 of 1 PASS (native cos min 0.99988, floor 0.99984) |

The first production build of this file failed under the build's
`-Werror=unused-variable`: 14 device-only constants are unused in the host pass.
They are marked `[[maybe_unused]]`; the incremental rebuild then succeeded
(`build_exit=0` after 174 s) and the file compiles alone in 42 s with the same
flags.

Each kernel test compares with an fp32 / numpy reference and can fail (see the
mutation checks below).  The real-image test needs the checkpoint under
`~/hfcache/inkling2-mini` and the transformers branch with `inkling2`.

### Mutation checks (by hand)

Each row is one source mutation of the kernel, built as a separate library and
run against the kernel tests with `-x` (stop at the first failure).

Kernel (22 mutations; the first failing test is shown):

| # | mutation of `inkling2_vit_rdna2.cu` | first failing test |
|---|---|---|
| m01 | attention softmax scale 0.125 -> 0.130 | `test_attention_matches_fp64_softmax[128-[(1, 1)]]` |
| m02 | key mask `< L` -> `<= L` (one key past the image) | `test_attention_matches_fp64_softmax[128-[(1, 1)]]` |
| m03 | V-pair permutation selector changed | `test_attention_matches_fp64_softmax[128-[(1, 1)]]` |
| m04 | GELU erf -> tanh form | `test_gemm_epilogues[2-1-128-128]` |
| m05 | LayerScale gamma dropped in the residual epilogue | `test_gemm_residual_accumulates_scaled_branch_in_fp32[1-128-128]` |
| m06 | RoPE patch position `px + 1` -> `px` | `test_rope_tables_and_q_k_rotation` |
| m07 | RoPE row / column axes swapped | `test_rope_tables_and_q_k_rotation` |
| m08 | RoPE rotation sign of k | `test_rope_tables_and_q_k_rotation` |
| m09 | position-embedding source clamp at 0 removed | survived the first run (67 passed: the old grids never reached the clamp); killed after adding the `grid=8` case: `test_embed_adds_bilinear_position_and_layernorms[8]` (the `grid=32` case still passes) |
| m10 | pool reads row `t - 1` | `test_pool_drops_each_images_cls_row` |
| m11 | LayerNorm epsilon dropped | survived the first run (67 passed); killed after adding the case `(9, 256, 1e-3, 0.0)` (variance far below the epsilon, zero mean): `test_layernorm_fp32_rows_to_fp16[9-256-0.001-0.0]`; the first version of the case (mean 1) failed on the unmutated kernel by 4.8e-4 and was changed, the kernel was not |
| m12 | `kF16Bias` bias dropped | `test_gemm_epilogues[1-1-128-128]` |
| m13 | `exact` gather: `255` -> `254` | `test_gather_patches_is_exact_for_every_raw_value[1-dtype0-0]` |
| m14 | GEMM live-row predicate `+ 1` | `test_gemm_epilogues[0-1-128-128]` |
| m15 | GEMM B-fragment dword swap removed | `test_gemm_epilogues[0-1-128-128]` |
| m16 | attention accumulator rescale `o *= alpha` removed | `test_attention_matches_fp64_softmax[128-[(16, 16)]]` |
| m17 | attention denominator rescale removed | `test_attention_matches_fp64_softmax[128-[(16, 16)]]` |
| m18 | `clip_bf16`: round-to-nearest-even -> truncation | `test_gather_patches_is_exact_for_every_raw_value[1-dtype0-1]` |
| m19 | `clip_bf16`: std of channels 1 and 2 swapped | `test_gather_patches_is_exact_for_every_raw_value[1-dtype0-1]` |
| m20 | `clip_bf16`: mean[0] 0.48145466 -> 0.4815 | `test_gather_patches_is_exact_for_every_raw_value[1-dtype0-1]` |
| m21 | gather channel index `(ch + 2) % 3` -> `(ch + 1) % 3` | `test_gather_patches_is_exact_for_every_raw_value[1-dtype0-1]` |
| m22 | gather mode dispatch swapped | `test_gather_patches_is_exact_for_every_raw_value[1-dtype0-0]` |

Processing (`processing.py`, disposable copy of the tree, the file
`test_vision_processing.py` run with `-x`; baseline 32 passed, restored copy 32
passed): all eight mutants fail.

| # | mutation | first failing test |
|---|---|---|
| P1 | padding `-1` -> `0` | `test_cells_are_row_major_with_bottom_right_padding` |
| P2 | grid rows and columns swapped | `test_grid_is_ceil_of_40_pixel_cells[39-41-1-2]` |
| P3 | checkerboard levels 204 / 153 swapped | `test_transparency_composited_on_checkerboard` |
| P4 | alpha blend rounding `+ 127` -> `+ 0` | `test_transparency_composited_on_checkerboard` |
| P5 | pad-behind-marker collapse tests the wrong token | `test_template_pad_behind_the_marker_is_not_doubled[tail1]` |
| P6 | marker token dropped from the replacement | `test_placeholder_expansion` |
| P7 | cell order transposed (`permute(0, 2, 3, 1, 4)`) | `test_cells_are_row_major_with_bottom_right_padding` |
| P8 | image-size guard loosened tenfold | `test_images_larger_than_one_encoder_launch_are_refused` |

## Wiring for the lead

`amd/model.py` (the text class is renamed, the served class is the composition;
the registry entry already points at `Inkling2LoopForConditionalGeneration`):

```diff
 from vllm.config import VllmConfig
+from vllm.model_executor.models.interfaces import SupportsMultiModal
 ...
+from .multimodal import Inkling2MultiModalMixin, register_inkling2_processor
 ...
-class Inkling2LoopForConditionalGeneration(nn.Module):
+class Inkling2LoopTextForCausalLM(nn.Module):
 ...
     def load_weights(...):
-            if name.startswith(("mtp.", "model.visual.")):
+            if name.startswith("mtp."):
                 continue
 ...
+@register_inkling2_processor
+class Inkling2LoopForConditionalGeneration(
+    Inkling2MultiModalMixin, Inkling2LoopTextForCausalLM, SupportsMultiModal
+):
+    """Text and image entry point of ``Inkling2LoopForConditionalGeneration``."""
```

The mixin's `__init__` builds the text class with `vllm_config=` / `prefix=`
(unchanged signature) and then the tower under `visual`; `load_weights` strips
`model.visual.` names into the tower and hands the rest to the text loader;
`embed_input_ids(input_ids, multimodal_embeddings=None, *, is_multimodal=None)`
wraps the text embedding.  `Inkling2MultiModalMixin.get_language_model()` returns
the text model itself.  No change is needed in `registry.py`
(`_TEXT_GENERATION_MODELS` already holds the entry, like `InklingForConditionalGeneration`).

Serving notes: the decoder tokens of one image (4056 for the 3082x2048 photo) must
fit the scheduler's encoder budget, so `--max-num-batched-tokens` has to be raised
above the largest image you intend to accept **[INFERENCE: the scheduler path was
not exercised in this work]**; the profiling run uses a dummy image of at most
2048x2048 (grid up to `[1, 52, 52]`, 2704 decoder tokens).

## Limits and open risks

* The tower is replicated on every TP rank (636 MB fp16 each) and runs eagerly;
  at TP=4 the same image is encoded four times.
* Latency is the encoder alone: 41 ms for a 640 px crop, 358 ms for a 1600 px
  crop, 1.67 s for the full 3082x2048 photo (all measured, one V620); it blocks
  the step that schedules the image.
* An image of more than 16 383 cells (65 535 ViT tokens, about 5120 x 5120 px) is
  refused with a `ValueError` by the processor (a request-level error) and by the
  encoder; it is not tiled.  Images of up to 16 225 ViT tokens were run.
* The pixel path (D6) is the largest numeric deviation and the training-time
  choice is **[INFERENCE]**; flip `pixel_norm` if the checkpoint turns out to
  want the exact formula.
* The fp16 floor (cos min 0.9999 in `clip_bf16` mode, 0.9990 to 0.9975 in `exact`
  mode for 6.4k to 16k ViT tokens) is well below the bf16 reference's own noise
  (cos min 0.995 and 0.975 for the 640 and 1600 px crops); no end-to-end
  text-quality benchmark with images was run in this work.
* The encoder is not captured in a HIP graph.  The mixin was tested against the
  calling contract of the V2 runner (`embed_multimodal`, `embed_input_ids`) and
  composed with the text stack on the tiny checkpoint; a served end to end run
  with an image request was not done in this work.
