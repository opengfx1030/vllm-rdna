# Silicon: RidgeFill on gfx1030

**Status: measured.** Path A (rocBLAS fp16) is the default. Path B (int8
`sdot4`) and an fp16 `fdot2` map kernel are compiled, opt-in and measured
in [TESTPLAN.md](TESTPLAN.md), item 4. Speedups quoted below are the
paper's; the gfx1030 numbers are in TESTPLAN.md.

Target: V620 / gfx1030, Wave32, `fdot2` / `sdot4` only (no WMMA, no MFMA,
no FP8). TP=4 over PCIe. For filled prefix rows,
`x_hat_l = A_l x_S + b_l` (paper sections 2.1–2.2). Late layers write their
cache from `x_hat_l` (attention K/V plus the Flash-Next indexer keys, and
the GDN recurrent update) and skip the MoE. Decode is exact.

gfx1030 has packed fp16 dot (`v_dot2_f32_f16`) and no bf16 dot.
`csrc/rocm/rdna/common/arch.cuh` sets `kHasDot2F16` for every RDNA generation
and `kHasDot2Bf16` false for RDNA2. `FeatureDot12Insts` (`v_dot2_f32_bf16`)
is not in the gfx1030 feature set; the monokernel note cites LLVM
`AMDGPU.td` for that. bf16 GEMMs on this GPU are expected to be slower.
`kHasFp8Wmma` is RDNA4 only. The
[W8A8 note](https://github.com/BlivionIaG/rdna-hip-wiki/blob/main/kernels/quant/w8a8.md)
says gfx1030 has no FP8 unit. The tcclaviger image and the MXFP4/FP8
checkpoints target RDNA4 (R9700, gfx1201; `vllm/platforms/rocm.py` device
`0x7551`). Do not run that image here. Port the method and refit.

## Ranked implementations

- **A, take, no new kernel.** Dequant the int8 maps to **fp16** at load and
  apply them with the existing dense GEMM (`F.linear`, rocBLAS / hipBLASLt),
  **fp32 accumulate**. bf16 is the fallback when a range check fails or
  `VLLM_RDNA_RIDGEFILL_MAP_DTYPE=bf16`. The public kit dequants to bf16
  (`code/c5/convert.py` writes `layer.L` `[2560, 10241]` bf16;
  `F.linear` is `Projector.__call__` in `code/kva/kva.py`, tag v1.8). That
  dtype matches their RDNA4 box, not gfx1030.
- **B, measured, opt-in.** `VLLM_RDNA_RIDGEFILL_MAP_FUSED=1` with
  `VLLM_RDNA_RIDGEFILL_MAP_MODE=dot4`: the maps are int8 per output row
  after a Hadamard-128 rotation of K, the filled rows are rotated and
  quantized per row on the fly, and `v_dot4_i32_i8` accumulates in int32
  (`csrc/rocm/ridgefill_map_rdna2.cu`). The paper's group-128 scales are
  not used: per row after the rotation is within 0.002-0.005 rel RMS of
  group-128 on the captured layers and keeps the epilogue to one multiply.
  On the map shape it runs at 26.8-29.5 T-MAC/s, 2.5-3.9x the per-layer
  rocBLAS; `MAP_MODE=dot2` is the same kernel on `fdot2`, 1.2-1.8x. In the
  compiled TP4 A/B (a413af6b52) dot4 cuts 16K solo TTFT by 8.3 % against
  the rocBLAS fill (4.86 s against 5.30 s) at equal PPL (+0.001); dot2 cuts
  1.8 %. The fill itself costs +0.72 long-window PPL against exact on every
  map path, so neither is a serving recommendation (TESTPLAN.md, item 4).
- `rdna_gemm_smallm` (M ≤ 32, decode) is not the map path. The filled row
  count is in the thousands.
- Late-layer projections from `x_hat` reuse the existing W4A16 / EXL3 /
  MXFP4 prefill GEMMs. The weights are unchanged (paper figure 2b). No new
  kernel, and none of those `csrc` files are edited by this draft.

## Map shape, dtype, VRAM (Flash-Next)

24 maps, layers 24–47. Each is int8 codes `[2560, 10240]`, bf16 scale
`[2560, 80]` (group 128), bf16 bias `[2560]`. Sources: the Hugging Face
repo, paper section 2.2, and the header of `code/c5/convert.py` at v1.8.

On disk, about 0.60 GiB (paper) or about 0.65 GiB on Hugging Face once
`correction.safetensors` and `rowsel.safetensors` are included. Path A's
fp16 weights, `24 × 2560 × 10240 × 2`, are about **1.17 GiB**. The bias
column the public convert appends (`10241`) does not change that at the
precision we quote.

Replicated, the default: about **1.17 GiB per GPU** at TP=4. The public
convert writes one full safetensors file, and `engine_check.py` (present at
`code/c5/engine_check.py` in v1.8) loads the full weight. Only the
terminal-state correction is head-sharded, and that shard is for their
TP=2 (`GDN_HEADS, TP = 48, 2` in `convert.py`). Their measured serve was
TP=2 (paper section 5.1). Our capture runs at TP=4/EP=4, the
`flashnext` serve shape, and the map is replicated on every rank. The TP
policy of the closed `ced.py` was not
visible; this box did not open the image (unsourced).

Sharding the output dimension would be about 0.29 GiB per GPU. Nothing in
the public `c5` artifacts shows that split. Do not do it for the first port.
The README's HC section is why a replicated map does **not** all-gather
`x_S`: the 10240-wide residual is already on every rank.

27B: 48 maps of `5120 × 5120` int8, about 1.2 GiB (paper section 2.2). The
same A/B choice. The released file is a GGUF for llama.cpp.

## fp16 range

The map reads the four raw, pre-norm streams entering layer 24. Those are
the large values. The paper does not say whether the fit target is pre-norm
or post-norm. Keep the input in fp32 (capture) or bf16 until both checks
pass:

- max abs of the 10240-wide input
- max abs of the 2560-wide fp32 prediction

Log both. If either exceeds 65504, that tensor stays bf16 and the GEMM for
that call is bf16. If both fit, cast to fp16 and run the fp16 GEMM with
fp32 accumulate. `math.apply_checked` is the CPU form of this decision.
The V620 gate is in `TESTPLAN.md`. It has not been run.

## GDN filled rows

Paper section 2.3: a filled GDN row still runs the projections and the
state update from `x_hat`, and skips the MoE. The error folds into the
recurrent state the tail inherits. Dest's HIP GDN prefill
(`gdn_prefill_{prep,kkt,solve_wy,delta_h,o}_rdna2`) already consumes
projected inputs per chunk
([gdn-prefill.md](https://github.com/BlivionIaG/rdna-hip-wiki/blob/main/kernels/linear-attn/gdn-prefill.md)).
No RidgeFill-specific GDN kernel. Feeding predicted inputs instead of exact
ones is engine glue. Do not edit those `csrc` files for this draft.

## Catches

- The map GEMM is large-M prefill. Decode and cudagraph capture of decode
  shapes are unchanged. Production graphs stay `FULL_AND_PIECEWISE`.
- Path A needs no custom LDS or `fdot2` occupancy work. Path B's tile was
  built from a profile: resource counts, dot rates and the tile sweep are
  in TESTPLAN.md, item 4.
- Replicated apply does not all-gather `x_S`. See the README. Validate on a
  real TP=4 boot anyway: log the boundary width (the draft raises if it is
  not 10240) and confirm the four ranks match.

## Attribution

Paper title: RidgeFill: Training-Free Late-Layer Cache Fill for Faster
Prefill, Dylan Johnston and Robert P. Smith. OpenReview `ymLpDkJIGs`,
Zenodo `10.5281/zenodo.23890247`. Speedups quoted from the paper are not
ours; our gfx1030 measurements are in TESTPLAN.md.
