# MonoKernel plan for gfx1030 (Flash-Next, FFN-only)

> **Current state (2026-10-16):** the MonoKernel ships behind `--monokernel`;
> usage, parts and knobs are in
> [`vllm/models/qwen4_exp/amd/mono/rdna2/README.md`](../../../vllm/models/qwen4_exp/amd/mono/rdna2/README.md),
> tools in [`tools/rdna/mono/`](../../../tools/rdna/mono/README.md). The pages
> in this directory are the history (plan, P0, 2a, Stage E); their
> `rdna_mono_*` keys, `mono_ffn.py` and `tools/monokernel-2a/` paths are the
> pre-move names.

Status: **plan**. Nothing here is wired into the vLLM build. There is no
`--monokernel` flag, no kernel under `csrc/`, and no CMake or Python behavior
change. The only code in this change is the standalone P0 probe under
[`tools/monokernel-p0-probe/`](../../../tools/monokernel-p0-probe/README.md).
That probe is **not yet compiled; first build on V620 with ROCm 7.14 may need
small fixes.**

Read against dest `rdna_extras` at
[`e245a469`](https://github.com/opengfx1030/vllm-rdna/commit/e245a4697a139fe8d3bbec4b9c837c37279674e9)
(2026-10-13). Phase order is in [plan.md](plan.md). Silicon notes that this
page relies on are the merged wiki plan
([`rdna-hip-wiki` #24](https://github.com/BlivionIaG/rdna-hip-wiki/pull/24),
merge [`9b91d411`](https://github.com/BlivionIaG/rdna-hip-wiki/commit/9b91d4116d24414ea2ffd9530d0caeed331ddcd0)).
Items this tree has not measured on a V620 are marked **(unverified)**.

## Goal

Port the upstream MonoKernel idea to gfx1030 on a 4× Radeon PRO V620 box,
tensor-parallel size 4, for Qwen3.8-Flash-Next decode (`qwen4_exp`).

A MonoKernel is one persistent kernel that runs most or all of one decoder
layer's decode step in a single launch. The
[RFC](https://github.com/vllm-project/vllm/issues/60904) asks for it off by
default behind `--monokernel`, decode-only, default-deny, Model Runner V2
only, with bounded waits, sticky error words, and fail-stop. Prefill and
mixed steps stay on vLLM's layers
([RFC, non-goals](https://github.com/vllm-project/vllm/issues/60904)).

The template is
[#61325](https://github.com/vllm-project/vllm/pull/61325): a draft Flash-Next
mono decode on MI355X (gfx950) at TP2, two persistent launches per layer
(K1, K2), pure decode, 1–8 rows, no spec decode. K1 uses up to 55 KB of LDS
and K2 up to 68 KB. The model class there is
`Qwen4ExpForConditionalGeneration`.

Two earlier ports fix the shape of the problem:

- [#60397](https://github.com/vllm-project/vllm/pull/60397) (merged
  2026-10-10) is DeepSeek-V4.1-Flash on MI355X. Thirty layers run K1+K2.
  Ten run an FFN launch after vLLM attention. TP all-reduces inside the
  kernel use symmetric peer memory. Hand-offs are epoch-tagged mailboxes in
  per-width scratch, so a FULL graph does not need a host clear.
- [#60731](https://github.com/vllm-project/vllm/pull/60731) ports that DSv4.1
  kernel to gfx942. A workgroup there has 64 KB of LDS, against 160 KB on
  gfx950, so the GEMV pass holds 6 tokens instead of 16 and the row cap is
  42.

[#61001](https://github.com/vllm-project/vllm/pull/61001) is a WIP draft
("will be closed") of `MonoOp` / `MonoSpec` / `MonoKernel` / `MonoRuntime`.
It allows an arch-specific spec and a kernel over an arbitrary slice of a
layer. This port does not adopt that API. It follows the RFC's MonoSpec and
safety rules, and it uses #60731's lesson that the row cap is per arch.

The production recipe on this fork is `flashnext-mtp2`: TP=4,
`VLLM_USE_V2_MODEL_RUNNER=1`, `CG_MODE=FULL_AND_PIECEWISE`, `MTP=2`,
`VLLM_RDNA_AR=1`
([`tools/rdna/recipes/flashnext-mtp2.env`](https://github.com/opengfx1030/vllm-rdna/blob/e245a4697a139fe8d3bbec4b9c837c37279674e9/tools/rdna/recipes/flashnext-mtp2.env)).
The launcher turns MTP=k into capture widths `(k+1)×{1,2,4,8}`
([`rdna_mtp_args`](https://github.com/opengfx1030/vllm-rdna/blob/e245a4697a139fe8d3bbec4b9c837c37279674e9/tools/rdna/rdna_launcher_common.sh#L123-L136)).
For k=2 that list is 3, 6, 12, 24. `CG_CAPTURE_SIZES=1` means "emit that
list", not "capture only size 1"
([`serve_rdna.sh`](https://github.com/opengfx1030/vllm-rdna/blob/e245a4697a139fe8d3bbec4b9c837c37279674e9/tools/rdna/serve_rdna.sh#L296)).
Whether a live server actually captures those widths is a P0 question
**(unverified)**.

## Take / leave

**Take** the RFC contract and its safety rules
([RFC](https://github.com/vllm-project/vllm/issues/60904)):

- One spec per model, off by default, one switch.
- Decode only. A step that is not pure decode calls the vLLM layer.
- Default-deny: a feature in use that `supports` does not list is refused
  at startup.
- Every cross-workgroup and cross-rank wait is bounded. Expiry sets a sticky
  device error word. Later waits exit early.
- The output rank does not return tokens from a step whose error word is set.
  Fail-stop exits the workers. A kernel does not disable itself and continue.
- The grid fits the CUs that are actually present. A partly resident grid
  that waits on its own peers can deadlock.
- Rank-uniform routing: the decision uses only metadata every rank holds.

**Leave** the FlyDSL / aiter MFMA and MXFP4 kernels. #61325 reads
`amd/Qwen3.8-Flash-Next-Quark-MXFP4` and AITER's MXFP4 expert layout. #60397
reads AITER's shuffled MXFP4 experts and MXFP8 attention linears. Those
instructions are not in the gfx1030 feature set (see Silicon). The RFC also
puts shared device primitives in FlyDSL
([RFC, kernel source](https://github.com/vllm-project/vllm/issues/60904)).
This port does not take that dependency.

## Quantization matrix v1

v1 quantizes **routed experts only**, and only as symmetric group-quantized
W4A16 GPTQ. The inner product is `v_dot2_f32_f16` (`fdot2`), which the
existing RDNA2 MoE kernel already uses
([`moe_q_gemm_rdna2.cu`](https://github.com/opengfx1030/vllm-rdna/blob/e245a4697a139fe8d3bbec4b9c837c37279674e9/csrc/rocm/moe_q_gemm_rdna2.cu#L27-L29)).
Scales are per group; zero points in that method are synthesized for the
symmetric bias, not loaded from an asymmetric checkpoint
([`compressed_tensors_moe_wna16_rdna2.py`](https://github.com/opengfx1030/vllm-rdna/blob/e245a4697a139fe8d3bbec4b9c837c37279674e9/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_wna16_rdna2.py#L1-L58)).

Router, shared expert, and gated residual stay **dense** (not quantized).
The checkpoint stores them as BF16 on disk, but **dest serves fp16**:
`rdna_fused_glue.cu` accepts `at::kHalf` only
([`check_weight`](https://github.com/opengfx1030/vllm-rdna/blob/e245a4697a139fe8d3bbec4b9c837c37279674e9/csrc/rocm/rdna_fused_glue.cu#L366-L373)), and the #81 TunableOp CSVs are
`GemmTunableOp_Half_*` rows only, no BFloat16
([`tunableop_results0.csv`](https://github.com/opengfx1030/vllm-rdna/blob/e245a4697a139fe8d3bbec4b9c837c37279674e9/bench_results/2026-09-29_tunableop-rows/curated/tunableop_results0.csv)). So on dest these dense parts use
`v_dot2_f32_f16` (`fdot2`) with fp32 accumulate, the same loop as
[`moe_resident_decode.cu`](https://github.com/opengfx1030/vllm-rdna/blob/e245a4697a139fe8d3bbec4b9c837c37279674e9/csrc/rocm/moe_resident_decode.cu#L83-L114)
([wiki plan, "Dense GEMV: dest fp16 path vs true bf16"](https://github.com/BlivionIaG/rdna-hip-wiki/blob/e889bbd3/kernels/persistent/monokernel-gfx1030-plan.md)).

The bf16 widen-then-scalar-`v_fma_f32` path is kept **only for a true bf16
deployment**, which dest is not. gfx1030's ISA set has neither
`v_dot2_f32_bf16` (`FeatureDot12Insts`) nor packed fp32 ops
(`FeaturePackedFP32Ops`)
([`AMDGPU.td` `FeatureISAVersion10_3_0`](https://github.com/llvm/llvm-project/blob/eb61ef62df4fd5151c4c379302f636ac430bc32d/llvm/lib/Target/AMDGPU/AMDGPU.td#L2268-L2282),
[`Dot12Insts`](https://github.com/llvm/llvm-project/blob/eb61ef62df4fd5151c4c379302f636ac430bc32d/llvm/lib/Target/AMDGPU/AMDGPU.td#L924-L926),
[`PackedFP32Ops`](https://github.com/llvm/llvm-project/blob/eb61ef62df4fd5151c4c379302f636ac430bc32d/llvm/lib/Target/AMDGPU/AMDGPU.td#L801-L803));
[`arch.cuh`](https://github.com/opengfx1030/vllm-rdna/blob/e245a4697a139fe8d3bbec4b9c837c37279674e9/csrc/rocm/rdna/common/arch.cuh#L53-L62)
records `kHasDot2Bf16` false for RDNA2. Casting true bf16 weights to fp16 to
reach fdot2 is avoided (fp16 range ±65504) ([wiki plan](https://github.com/BlivionIaG/rdna-hip-wiki/blob/e889bbd3/kernels/persistent/monokernel-gfx1030-plan.md)).

**Router numerics.** Router logits come out in fp32. Sigmoid/softmax, bias
correction and top-k run **once** in fp32, and the combine reuses the same
ids and weights ([wiki plan](https://github.com/BlivionIaG/rdna-hip-wiki/blob/e889bbd3/kernels/persistent/monokernel-gfx1030-plan.md)).

A V620 microbench of that GEMV is **(unverified)**.

### Out of v1

| Format | Why it is out |
| --- | --- |
| W4A8 / `sdot4` | Dest's dense W4A8 arm runs only at `M >= 33` ([`W4A8_MIN_ROWS`](https://github.com/opengfx1030/vllm-rdna/blob/e245a4697a139fe8d3bbec4b9c837c37279674e9/vllm/model_executor/kernels/linear/mixed_precision/rdna2_w4a16.py#L45-L49)). Dest's MoE W4A8 arm runs only at `>= 512` tokens, because below that the W4A16 kernel was as fast or faster on Flash-Next shapes ([`W4A8_MOE_MIN_TOKENS`](https://github.com/opengfx1030/vllm-rdna/blob/e245a4697a139fe8d3bbec4b9c837c37279674e9/vllm/model_executor/layers/fused_moe/experts/rdna2_w4a16_moe.py#L41-L47)). This kernel's decode cap is 24 rows (batch 8 × 3 query rows at MTP k=2; see Gates). Both existing cutoffs sit above that cap, so today's W4A8 dispatch would not select those kernels. Revisit only if int8 activations are shown to pay at decode sizes **(unverified)**. That concerns the kernels, not a refusal: `--monokernel` runs next to `VLLM_RDNA2_W4A8_SDOT4=1`. Load leaves the same packed weights, scales and zeros with W4A8 on or off (it only sets a Python flag), K2 binds them as they are, the modules serve 9-24 rows on the W4A16 GEMM, and the W4A8 GEMM runs from 512 tokens. Only the expert cache refuses W4A8. |
| Asymmetric zero points | The native MoE path synthesizes symmetric zeros and does not load the checkpoint's ([`_synthesize_qzeros`](https://github.com/opengfx1030/vllm-rdna/blob/e245a4697a139fe8d3bbec4b9c837c37279674e9/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_wna16_rdna2.py#L44-L58)). [`is_supported`](https://github.com/opengfx1030/vllm-rdna/blob/e245a4697a139fe8d3bbec4b9c837c37279674e9/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/rocm_moe_rdna.py#L16-L31) returns false for `symmetric=False` and the checkpoint falls through to Triton WNA16. That is dest [#84](https://github.com/opengfx1030/vllm-rdna/pull/84). Out until a native kernel honors real zero points. |
| MXFP4, MXFP8, FP8 | #61325 and #60397 use MFMA-class matrix ops and MX layouts on gfx950. `FeatureISAVersion10_3_0` does not include `FeatureMAIInsts`, `FeatureFP8Insts`, `FeatureFP8ConversionInsts`, or `FeatureDot11Insts` (the fp8 dot4 ops). Those features are defined in [`AMDGPU.td`](https://github.com/llvm/llvm-project/blob/eb61ef62df4fd5151c4c379302f636ac430bc32d/llvm/lib/Target/AMDGPU/AMDGPU.td#L921-L942) and are not in the [10.3.0 set](https://github.com/llvm/llvm-project/blob/eb61ef62df4fd5151c4c379302f636ac430bc32d/llvm/lib/Target/AMDGPU/AMDGPU.td#L2268-L2282). gfx1030 has no FP8 conversion instructions in that feature set. |
| EXL3 | Trellis decode is its own kernel family ([`exl3_dot2_common.cuh`](https://github.com/opengfx1030/vllm-rdna/blob/e245a4697a139fe8d3bbec4b9c837c37279674e9/csrc/rocm/exl3_dot2_common.cuh#L1-L23)), with codebooks `3inst` / `mcg` / `mul1` ([`exl3.py`](https://github.com/opengfx1030/vllm-rdna/blob/e245a4697a139fe8d3bbec4b9c837c37279674e9/vllm/model_executor/layers/quantization/exl3.py#L132-L155)). The in-tree 27B bench names production codebook `mul1` ([`bench_results/2026-10-01_exl3-27b-mul1-tp4/SUMMARY.md`](https://github.com/opengfx1030/vllm-rdna/blob/e245a4697a139fe8d3bbec4b9c837c37279674e9/bench_results/2026-10-01_exl3-27b-mul1-tp4/SUMMARY.md)). Later candidate, not v1. |

## Refusal

Refusal looks at **which tensors the checkpoint quantized**, not only the
format name. If the router, the shared expert, or the gated residual is
quantized, startup refuses the MonoKernel for that layer and the step runs
vLLM's layers. A symmetric W4A16 GPTQ checkpoint whose routed experts are
the only quantized GEMMs is the v1 shape. Anything else is a refuse.

Also out of the kernel, and left on vLLM's layers:

- The INT4 PLE n-gram table. Dest [#49](https://github.com/opengfx1030/vllm-rdna/pull/49) serves that table through the upstream PLE lookup (`VLLM_PLE_QUANT_DIR`). It is not a GEMM this kernel replaces.
- MTP drafter experts produced by the quantizer bundled in [#82](https://github.com/opengfx1030/vllm-rdna/pull/82) (`tools/rdna/quant_mtp_experts.py`). K2 still needs a supported symmetric W4A16 drafter; K1 can independently fuse compatible attention HyperConnections. #60397 leaves its DSpark draft layers off the main K1+K2 path ([#60397, scope](https://github.com/vllm-project/vllm/pull/60397)).

### Experimental MTP drafter routing

The `--monokernel` path can be opted into for a compatible Qwen4Exp MTP
drafter with `additional_config.rocm_mono_decode.mtp_drafter=true`. This is
separate from the target-model K1/K2 switches and defaults to off. Drafter K1
and K2 have independent row caps (`mtp_k1_max_rows`, `mtp_k2_max_rows`, each
clamped to 24). K1 setup depends only on compatible MLP and attention
HyperConnection layouts, so it can run when the drafter's dense FFN is
unsupported by K2. K2 still requires the supported symmetric W4A16 expert
layout. Each part falls back to its existing vLLM modules when its own layout,
dtype, row cap, or graph mode is unsupported. The drafter uses its own
persistent-kernel workspace so its sticky error state cannot latch the target
model's workspace.

This is an experimental routing hook, not a mixed-prefill safety or performance
qualification. The row cap only provides a modules fallback for large batches;
it cannot rule out GPU scheduling delays on a small persistent-kernel launch.
The K1/K2 timeout path writes NaNs to the caller-visible outputs and latches
its workspace. Opt-in MTP launches enqueue a same-stream async memset of the
small control header before the persistent kernel; this is captured as a graph
memset node, not an extra HIP kernel launch. It clears a previous sticky error
so a later request does not inherit it, while the current failed invocation
still propagates NaNs. Enabling this knob requires
`VLLM_RAISE_ON_LOGIT_NANS=1`; startup refuses the MTP route otherwise. This
makes a detected NaN fail loudly instead of silently feeding corrupted logits
to sampling, at the cost of the runner's logits NaN scan. Neither the reset nor
the fail-stop proves that a barrier cannot time out or guarantees worker
recovery. Keep the regular target MTP2 route unchanged. MTP MonoKernel is an
experimental performance path until mixed-load fault/recovery and c1/c8
serving comparisons show a win beyond run variance; the reset node's latency
must be included in those measurements.

Startup still applies the RFC default-deny list (PP, DP, EP, LoRA, KV
connectors, sleep mode, and the rest named in the
[RFC](https://github.com/vllm-project/vllm/issues/60904)) for any feature the
gfx1030 spec does not list. That list is the contract. Which of those the
first gfx1030 spec will set in `supports` is decided with the spec, not in
this page.

## Silicon

| Choice | Source |
| --- | --- |
| 72 workgroups × 256 threads, one workgroup per CU | V620 is Navi 21 with 72 CUs ([Wikipedia, Radeon Pro V620 row](https://en.wikipedia.org/wiki/Radeon_Pro), citing AMD's V620 page). The P0 probe launches `grid = 72`, `block = 256` ([`probe.hip`](../../../tools/monokernel-p0-probe/probe.hip)). |
| 64 KiB LDS per workgroup | In-tree RDNA trait `kLdsBytes = 64 * 1024`, commented as the per-workgroup budget in WGP mode ([`arch.cuh`](https://github.com/opengfx1030/vllm-rdna/blob/e245a4697a139fe8d3bbec4b9c837c37279674e9/csrc/rocm/rdna/common/arch.cuh#L61-L62)). The gfx10.3 skinny GEMV uses the same cap ([`skinny_gemms_int4.cu`](https://github.com/opengfx1030/vllm-rdna/blob/e245a4697a139fe8d3bbec4b9c837c37279674e9/csrc/rocm/skinny_gemms_int4.cu#L15-L21)). |
| Re-tile #61325's K2 | #61325 reports K2 up to 68 KB of LDS on gfx950, which is over this 64 KiB cap. A stage that needs ~68 KB has to be re-tiled before it can launch. |
| WGP vs `-mcumode` | The probe builds the default (WGP) binary and a `-mcumode` binary ([`build.sh`](../../../tools/monokernel-p0-probe/build.sh)). It prints `.amdhsa_workgroup_processor_mode` from the code object. The README's expected values (1 = WGP, 0 = CU mode) are **(unverified)** until the first V620 run. The probe also says, unmeasured, that 64 KiB LDS is 1 block/CU in CU mode and 2 blocks per WGP if the WGP has 128 KiB **(unverified)**. |
| Cross-workgroup flags | Agent-scope release store and acquire load ([`probe.hip` `bounded_wait`](../../../tools/monokernel-p0-probe/probe.hip)). The RFC requires a release fence before further data and says the remote-store ordering of that fence is not formally guaranteed ([RFC, safety](https://github.com/vllm-project/vllm/issues/60904)). |
| `s_sleep` in spins | The probe's wait calls `__builtin_amdgcn_s_sleep(2)` between polls ([`probe.hip`](../../../tools/monokernel-p0-probe/probe.hip)). |
| Bounded waits, phase id | On timeout the probe writes `{phase, wg, peer, observed, expected}` into a sticky record ([`StickyErr`](../../../tools/monokernel-p0-probe/probe.hip)). Dest's all-reduce already encodes a phase in its abort word ([`rdna_allreduce.cuh`](https://github.com/opengfx1030/vllm-rdna/blob/e245a4697a139fe8d3bbec4b9c837c37279674e9/csrc/rocm/rdna_allreduce.cuh#L72-L96)). Phase-id logging for the in-kernel all-reduce is a 2b deliverable, not part of 2a. |

The RFC's residency rule is why the grid is one workgroup per CU and why the
probe's progress test launches a busy kernel on the other stream. Whether
plain launch or `hipLaunchCooperativeKernel` stays resident under graph
replay is the P0 measurement **(unverified)**. The wiki plan recommends
epoch-tagged point-to-point flags, the same pattern as #60397's mailboxes,
if cooperative launch cannot be captured
([wiki plan](https://github.com/BlivionIaG/rdna-hip-wiki/blob/9b91d4116d24414ea2ffd9530d0caeed331ddcd0/kernels/persistent/monokernel-gfx1030-plan.md)).
This page does not choose between them before that run.

## Gates

Detail and the order of work are in [plan.md](plan.md). Summary:

1. **P0.** (a) The standalone probe: graph capture of the persistent kernel
   as a plain launch and as a cooperative launch; residency from code-object
   metadata with and without `-mcumode`; forward progress with busy-first
   and flag-first launch order. (b) Model Runner V2 questions, unanswered
   here **(unverified)**: does dest serve Flash-Next at TP4 under V2; does
   `FULL_AND_PIECEWISE` capture the widths above; does GDN / mamba-align
   refuse V2; how many launches a decode step issues today; and does this
   port need V2 at all. The RFC requires Model Runner V2
   ([RFC](https://github.com/vllm-project/vllm/issues/60904)). #61325's
   description does not mention it; that PR picks the path in the uncompiled
   outer `forward`
   ([#61325](https://github.com/vllm-project/vllm/pull/61325)). The recipe
   sets `VLLM_USE_V2_MODEL_RUNNER=1`. A search of `vllm/models/qwen4_exp/` at
   `e245a469` found no startup string that refuses V2. That is not a serve
   log.
2. **2a.** An FFN-only kernel: norm, router, shared expert, routed experts.
   vLLM's existing all-reduce runs on both sides. `VLLM_RDNA_AR` stays as
   the recipe sets it
   ([tuning guide](https://github.com/opengfx1030/vllm-rdna/blob/e245a4697a139fe8d3bbec4b9c837c37279674e9/docs/rdna2/tuning.md)).
   Attention, GDN, conv, and QSA stay on the current kernels.
3. **2b.** Fold both all-reduces into the kernel, using the
   `rdna_allreduce` protocol (uncached staging, epoch / sequence flags,
   sticky abort with a phase code;
   [`rdna_allreduce.cuh`](https://github.com/opengfx1030/vllm-rdna/blob/e245a4697a139fe8d3bbec4b9c837c37279674e9/csrc/rocm/rdna_allreduce.cuh#L1-L33)).
   Only if 2a gains time per output token and Blivion approves.
4. **3.** Time per output token at batch 1, 4, and 8 with MTP off, and at
   widths 3, 12, and 24 under `flashnext-mtp2` (k=2). GSM8K parity against
   the same build with the kernel off. The RFC's accuracy bar is full GSM8K
   (1,319 questions) plus a paired comparison
   ([RFC, report](https://github.com/vllm-project/vllm/issues/60904)).
   Plus a long-context KL check against the BF16 reference, because residual
   and gated-residual activations can exceed the fp16 max (±65504)
   ([wiki plan](https://github.com/BlivionIaG/rdna-hip-wiki/blob/e889bbd3/kernels/persistent/monokernel-gfx1030-plan.md)); see `plan.md` phase 3 for how to measure it.
   No number in that cell exists yet **(unverified)**.
5. **4.** One spec per layer type. #61325 describes Flash-Next as 48 decoder
   layers, 36 GDN and 12 QSA, with a QSA indexer on every fourth layer
   ([#61325](https://github.com/vllm-project/vllm/pull/61325)). This tree's
   `qwen4_exp` config takes `layer_types` from the checkpoint
   ([`config.py`](https://github.com/opengfx1030/vllm-rdna/blob/e245a4697a139fe8d3bbec4b9c837c37279674e9/vllm/models/qwen4_exp/config.py#L25-L26));
   the 36/12 split on the AWQ checkpoint is **(unverified)** until counted
   from that config. GDN recurrent state is kept FP32 and separate from the
   QSA heap (KV, compressed keys, indexer), which is the contract in the
   [wiki plan](https://github.com/BlivionIaG/rdna-hip-wiki/blob/9b91d4116d24414ea2ffd9530d0caeed331ddcd0/kernels/persistent/monokernel-gfx1030-plan.md).
   The HIP GDN decode kernel accepts either fp32 or fp16 state storage
   ([`gdn_decode_rdna2.cu`](https://github.com/opengfx1030/vllm-rdna/blob/e245a4697a139fe8d3bbec4b9c837c37279674e9/csrc/rocm/gdn_decode_rdna2.cu#L202-L206));
   which dtype the live server allocates is **(unverified)**. The MTP
   drafter stays on vLLM's layers.

## Build and run

Environment the probe is written for: ROCm 7.14, `--offload-arch=gfx1030`,
on a V620. Flags are pinned in
[`build.sh`](../../../tools/monokernel-p0-probe/build.sh):
`--offload-arch=gfx1030 -O3 -ffp-contract=fast -std=c++17`, no `-ffast-math`.

**Not yet compiled; first build on V620 with ROCm 7.14 may need small fixes.**

From `tools/monokernel-p0-probe/` (drafter: ROCM_specialist):

```bash
./build.sh
./run.sh
```

`build.sh` writes `build/wgp/probe` (default WGP mode) and
`build/cumode/probe` (`-mcumode`), plus code-object notes.

`run.sh` runs residency and graph capture for both binaries, then the
progress test at `--order busy-first` and `--order flag-first`. Each
invocation is its own process under `timeout -s KILL 300`
([`run.sh`](../../../tools/monokernel-p0-probe/run.sh)). Probe flags include
`--order busy-first|flag-first` and `--gap-ms` (host sleep between the two
launches). Extra arguments to `run.sh` are forwarded.

What the log is for, and how to read a stall, is in the
[probe README](../../../tools/monokernel-p0-probe/README.md).

Serve flag, **not implemented**:

```bash
vllm serve <checkpoint> --monokernel   # does not exist on this branch
```

The RFC's switch is `--monokernel`, default off
([RFC](https://github.com/vllm-project/vllm/issues/60904)). #61325 and #60397
use `VLLM_ROCM_MONO_DECODE=1` instead. This fork will follow the RFC
name when the flag is added. Until then a server started from this branch
has no mono path.
