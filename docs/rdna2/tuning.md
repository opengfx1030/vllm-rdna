# RDNA tuning guide: fork knobs, modes, TunableOp

What the fork adds on top of upstream vLLM for RDNA2 (gfx1030, primary) and
RDNA3 (gfx11, secondary), and what to turn on or off when tuning a serve.
Defaults below are the code defaults. A recipe may override them, so
`PRINT=1 bash tools/rdna/serve_rdna.sh RECIPE=<name>` shows what a launch
actually sets.

## Ground rules

- **Start from a recipe** (`tools/rdna/recipes/*.env`). Change one knob at a
  time with `KEY=value` on the launcher command line. Precedence:
  recipe < caller environment < trailing `KEY=value`.
- **Prefer CLI/config flags over env vars.** Config flags propagate to the
  engine-core and worker processes. Env-only knobs need checking: confirm in
  the serve log that the path took effect (each RDNA path logs when it engages).
- **Production is `FULL_AND_PIECEWISE` cudagraphs at full prefill and decode
  speed.** Eager (`EAGER=1`) must stay correct but is not a tuning target.
- **A/B before keeping a change**: run `tools/rdna/port_v031/serve_validate.sh`
  on both arms. Report prefill tok/s, decode tok/s per request, TTFT and ITL
  for 1k/512 and 16k/1k at c=1 and c=8. Greedy probe, perplexity, prefix and
  concurrency gates must pass on both arms.
- **RDNA3**: most HIP kernels are gated on `on_gfx10x()` and fall back to the
  upstream/Triton path on gfx11. Nothing here is validated on RDNA3 yet, and
  the shipped TunableOp rows are for one gfx1030 rocBLAS build only.

## Fork CLI / config parameters

| Flag | Default | What it does |
|---|---|---|
| `--decode-stall-budget-ms` | None (RDNA: `VLLM_RDNA_DECODE_STALL_MS`) | Decode-stall cap: bounds prefill tokens in steps that carry decodes, so a long prefill cannot freeze decoders for seconds. 0 disables. |
| `--decode-time-share` | 0.0 | With the cap, the minimum share of wall time kept for pure-decode steps while a prefill runs. |
| `--prefill-schedule-interval N` | 1 (off) | Admit prefills only every Nth step, so decoders get pure-decode steps (and MTP drafts). Upstream only applied this to data-parallel; the fork applies it to a single engine too. |
| `--linear-backend rdna2\|rdna_hybrid` | auto | Force the W4A16 linear kernel: `rdna2` = HIP decode+prefill; `rdna_hybrid` = skinny HIP + Triton (gfx10 dequant + rocBLAS at M ≥ 256). |
| `--attention-backend RDNA_ATTN` | auto | FA-RDNA2 attention (gfx10x). The launcher's `ATTN=fa` sets it. |
| `--speculative-config '{"method":"mtp","num_speculative_tokens":2,"use_local_argmax_reduction":true}'` | — | MTP=2. `use_local_argmax_reduction` avoids the drafter's vocab all-gather (required on TP=4). |
| `--monokernel` | off | Qwen4Exp (Flash-Next) decode on the RDNA2 MonoKernel: K1 (HC preamble, <= 24 rows), K2 (FFN block, <= 8 rows), fused GDN core, QSA decode, all on. `VLLM_ROCM_MONO_DECODE=1` is an alias (read in the API server). Knobs: `--additional-config '{"rocm_mono_decode":{"k2_max_rows":8,"gdn":true,...}}'`, only read with the flag. Needs `GPU_MAX_HW_QUEUES=4` (the Flash-Next recipes set it). Unsupported deployments refuse at startup (`--monokernel <reason>.`). See `vllm/models/qwen4_exp/amd/mono/rdna2/README.md`. |

## Launcher knobs (`tools/rdna/serve_rdna.sh`)

| Knob | Values | Effect |
|---|---|---|
| `MTP` | 0, 2 | Speculative tokens. Capture ladder becomes `(MTP+1)×{1,2,4,8}`. Keep powers of two (`CG_SIZES` overrides); 9/15/18/21 GPU-fault the 16k cell. |
| `ATTN` | fa, triton, none | FA-RDNA2 (`RDNA_ATTN` + GQA decode) / Triton AMD FA / leave to vLLM. |
| `CG_MODE`, `COMPILE_MODE`, `CG_EXTRA`, `EAGER` | | Cudagraph mode, compile mode, extra compilation JSON, eager fallback. |
| `PLE` | bf16, int4, int4-disk | Flash-Next PLE table, see [PLE modes](#ple-table-flash-next). Needs `PLE_QUANT_DIR` for int4*. |
| `W4A8` | 0, 1 | → `VLLM_RDNA2_W4A8_SDOT4`. |
| `RDNA_AR` | 0, 1 | → `VLLM_RDNA_AR` when the recipe leaves it unset (the flashnext/27B recipes set it to 1; override with `VLLM_RDNA_AR=0`). |
| `P2P` | 1, 0 | `0` = no PCIe peer-to-peer: `VLLM_RDNA_AR=0` (overrides the recipe) + `NCCL_P2P_DISABLE=1` (RCCL over host shared memory). For hosts where P2P is broken, and to bisect wrong output. Slower decode. |
| `TUNABLEOP`, `TUNABLEOP_PROFILE` | 1 / name | TunableOp lookup on/off, force a named profile. |
| `TUNED_CONFIG` | 0, 1 | Use `tuned-moe/` MoE configs (`VLLM_TUNED_CONFIG_FOLDER`). |
| `SEQS`, `MAXBAT`, `MAXLEN`, `KV`, `GMEM`, `BLOCK_SIZE`, `LPTH`, `PREFILL_INTERVAL` | | `--max-num-seqs`, `--max-num-batched-tokens`, `--max-model-len`, `--kv-cache-memory-bytes`, `--gpu-memory-utilization`, `--block-size`, `--long-prefill-token-threshold`, `--prefill-schedule-interval`. `KV=auto` (needs `GMEM`) drops `--kv-cache-memory-bytes` so vLLM sizes the cache after profiling. |
| `PP`, `PP_PARTITION` | | `--pipeline-parallel-size`; layers per stage → `VLLM_PP_LAYER_PARTITION`. See [Pipeline parallel](#pipeline-parallel-3-or-4-cards). |
| `ALLOC_CONF` | true, false, auto | `expandable_segments`. `auto` turns it off with the PLE sidecar. |
| `FEATURES` | tokens | `prefix-caching mamba-align expert-parallel language-model-only skip-mm-profiling vision-cap trust-remote-code reasoning auto-tool prompt-tokens-details`. |
| `HIP_VISIBLE_DEVICES` / `GPUIDS` | | GPUs. Keep a TP group on one PCIe switch. |

## Modes by subsystem

### Dynamic MTP graph capture

Target verification and draft prefill consume scheduled verification tokens;
subsequent autoregressive draft steps consume one token per request. Applying
the target schedule to those steps can generate negative capture widths and
fail `InputBatch.make_dummy` during graph-memory profiling.

With eight sequences, MTP=4 and schedule
`[[1,1,4],[2,2,2],[3,4,1],[5,256,0]]`, steady verification token totals
are 5, 6, 6, 8, 5, 6, 7, 8. A dense 1–8 ladder covers those shapes, but
does not cover large prefill or mixed batches. Bound uniform FULL captures
independently of a larger PIECEWISE ladder; keep mixed/varlen fallbacks.

**Capture coverage is not reliability certification.** Dynamic MTP also needs
accepted-state handoff and matching zero-draft PLE metadata at capture/replay.
The bounded-only image faulted under concurrency; do not omit those fixes.
For the tested TP8/eight-sequence schedule, the best concurrent ladder was
1–8 plus 64, 128, 256, 512, 1024, 1032 and 2048 rows (2.28–2.38 GiB/rank).
The 1–8-only alternative saves ~1 GiB/rank but loses concurrent decode speed.
All three passed correctness gates; 54 benchmark cells informed selection.
Revalidate prefix reuse, churn and repetition; this is not long-soak certification.
QSA ring alignment requirements remain independent of graph sizes.

### Attention

| Env | Default | Notes |
|---|---|---|
| `VLLM_USE_RDNA2_FA` | 0 | 1 pins `RDNA_ATTN` (FA-RDNA2) on gfx10x when no backend is set. |
| `VLLM_FA_RDNA2_GQA_DECODE` | 0 (launcher `ATTN=fa`: 1) | GQA-grouped decode kernel. |
| `VLLM_FA_RDNA2_GQA_MODE` | subgroup | GQA prefill mode. `subgroup` was measured faster than varlen/splitk. |
| `VLLM_FA_RDNA2_SPLIT_DECODE` | 1 | Split-K decode for MTP-verify and mixed batches (gated to ≤ 256 query×head rows). |
| `VLLM_FA_RDNA2_VERIFY_FULL_GRAPH` | 1 | Capture MTP-verify steps as FULL graphs (`UNIFORM_BATCH`). |
| `VLLM_FARDNA2_ENABLE_SPEC_GATE`, `VLLM_FARDNA2_SPEC_VERIFY_Q_LEN` | 0, 3 | Legacy gate that rejects verify batches. Leave off. |
| `VLLM_USE_RDNA2_MLA` | 0 | HIP sparse-MLA decode (DeepSeek-V4) instead of Triton. |

### Linear GEMMs (W4A16 / W4A8 / dense fp16)

| Env | Default | Notes |
|---|---|---|
| `VLLM_RDNA2_W4A8_SDOT4` | 0 | **Opt-in.** int8-activation prefill GEMMs; decode stays W4A16. Dense from M ≥ 33 (27B AWQ: prefill +24–34 %, PPL 4.90 vs 4.94); MoE from 512 tokens (Flash-Next: prefill +14–17 %, PPL 4.84 vs 4.86). `tools/rdna/bench_moe_w4a8.py` A/Bs the MoE kernel. |
| `VLLM_RDNA2_W4A16_RUNTIME_DISPATCH` | 0 | Choose decode/prefill kernel per call at run time (custom op) instead of at trace time. |
| `VLLM_RDNA_DENSE_INT8`, `_ONLY`, `_MIN_ROWS` | 0, 0, 64 | int8 weight-only shadow copies of dense fp16 projections for decode. Off in all recipes. |
| `VLLM_RDNA_DENSE_GEMV` | 0 | Force `gemv_f16_rdna2` for n ≤ 5. **Never route anything to wvSplitK on gfx1030**: it asserts on device. |
| `VLLM_DISABLED_KERNELS` | — | Upstream registry: e.g. `RDNA2W4A16LinearKernel` to fall back to Triton/Exllama. |

### MoE

| Env | Default | Notes |
|---|---|---|
| `VLLM_RDNA2_MOE_FP32_ACCUM` | 0 | 0 = packed-fp16 CAS epilogue (faster, run-to-run ULP flips). 1 = fp32 scratch (deterministic, for debugging/A-B). |
| `VLLM_RDNA2_MOE_BSM_LARGE`, `_BSM_MIN_TOKENS` | 8, 512 | Prefill tile width (rows) and the token count where it kicks in. 0 keeps the decode tile. |
| `VLLM_ROCM_MOE_SKINNY`, `_MAX_M` | 1, 8 | Wave-per-row skinny GEMV for **Triton WNA16** MoE weights only (max 16). Not used by the shuffled RDNA2 MoE path. |
| `VLLM_RDNA_MOE_RESIDENT`, `_SKINNY` | 0 | Keep RDNA2 W4A16 MoE weights in the native shuffled layout. Disables W4A8 MoE. |
| `VLLM_RDNA_FUSED_SE` | 1 | Fused shared-expert path (Qwen2/3-MoE family). |

### Hybrid models (GDN, conv1d, Qwen4Exp hyper-connection, QSA, PLE conv)

| Env | Default | Notes |
|---|---|---|
| `VLLM_GDN_DECODE_RDNA2` | 1 | HIP GDN decode. |
| `VLLM_GDN_HIP_PREFILL` | 0 | HIP GDN prefill prep. Off in recipes. |
| `VLLM_RDNA_GDN_OPAQUE` | 1 | Whole GDN layer as one opaque op. 0 keeps the projections in the compiled graph. |
| `VLLM_CAUSAL_CONV1D_RDNA2_FWD`, `_UPDATE` | 1 | HIP causal conv1d. Recipes set 0 (Triton). |
| `VLLM_RDNA_FUSED_HC` | 1 | Fused hyper-connection decode (Flash-Next). |
| `VLLM_RDNA_HC_PREFILL_FUSED` | 2 | HC prefill: 0 torch+Triton, 1 fused mix kernel, 2 split-K HIP down + fused kernel. |
| `VLLM_RDNA_HC_PAD_M` | 64 | Pad HC row counts to the 64-step grid that has TunableOp rows. 0 disables (and loses ~60 ms per 512-token mixed step). |
| `VLLM_RDNA_HC_PREFILL_HIP`, `VLLM_RDNA_PLE_CONV_HIP` | 0 | Opt-in HIP ports, not end-to-end verified. |
| `VLLM_RDNA_QSA_HIP` | 0 | HIP QSA path. Fault-free but 4–10 % slower than Triton; keep off. |

### All-reduce (TP over PCIe)

| Env | Default | Notes |
|---|---|---|
| `VLLM_RDNA_AR` | 0 (recipes 1) | RDNA one-shot push all-reduce for small messages, RCCL above. Check the log for `rdna_ar: … active` and no `rdna_ar: disabled`. |
| `VLLM_RDNA_AR_MAX_KB`, `_ONESHOT_KB` | 64, 64 | Size gate. **Do not raise**: two-shot is not boot-safe under PCIe load. |
| `VLLM_RDNA_AR_ALGO` | auto | `twoshot` forces two-shot (testing only). |
| `VLLM_RDNA_AR_BLOCKS`, `_PACE`, `_WAIT_MS` | auto, 0, — | Block cap, pacing, and peer-flag wait timeout. |
| `VLLM_FORCE_CUSTOM_ALL_REDUCE` | 0 | Stock custom AR past the XGMI gates. **Keep 0**: it has crashed the host (PCI SERR). |
| `NCCL_P2P_LEVEL=pxb`, `NCCL_PROTO=Simple`, `RCCL_P2P_NET_DISABLE=1`, `RCCL_P2P_BATCH_ENABLE=1`, `RCCL_MSCCL_ENABLE=0` | | RCCL settings for RDNA PCIe. LL/LL128 can deadlock. |
| `GPU_MAX_HW_QUEUES` | launcher 2 (flashnext TP4 recipes 4) | HIP streams per process; RDNA2 has 8 HQDs per GPU. Flash-Next TP4: 4 is neutral with the fork kernels off (MTP0/MTP2, 1k/16k, c=1/c=8; no "Runlist is getting oversubscribed") and fixes a per-rank skew from queue sharing at 2 when the persistent `rdna_mono_*` kernels are in the graph (2026-10-16). Other recipes keep 2 until measured. |

### Scheduler

| Env | Default | Notes |
|---|---|---|
| `VLLM_RDNA_DECODE_STALL_MS` | 0 | Default for `--decode-stall-budget-ms` on RDNA. Dense TP=4 recipes: 250 (max decode gap during 16k prefill 2672 → 320 ms, −8 % c=8 output). Opt-in on Flash-Next (−13 %), off at TP=2. |
| `VLLM_RDNA_DECODE_SHARE` | 0 | Default for `--decode-time-share`. |
| `VLLM_DECODE_STALL_MAX_OVERHEAD` | 0.25 | Floor on the cap so per-step fixed cost stays below this fraction of per-token work. |
| `VLLM_DECODE_STALL_FIT_ALL` | auto | Fit step cost on all prefill steps (dense default) or mixed steps only (MoE default). |
| `VLLM_RDNA_DYNAMIC_PREFILL` | 0 | Experimental live tuning of chunk cap and cadence. |

### Pipeline parallel (3 or 4 cards)

Flash-Next has nothing that divides by 3 (heads, experts, `moe_intermediate`),
so three cards means PP=3 at TP=1. Recipes `flashnext-pp3` (MTP0) and
`flashnext-pp3-mtp2` follow
[alanoo81/flashnext-v620-pp3](https://github.com/alanoo81/flashnext-v620-pp3):
text-only, 262K, KV sized from free memory, int4 PLE.

- **Step budget vs the Mamba block.** With prefix caching in `align` mode the
  block is 2048 tokens and prefill chunks are whole blocks. When decode tokens
  share the step, `MAXBAT` must exceed the block by the decode tokens of a full
  batch, or admission stalls (16K c=4 at MTP2: 3 running + 1 waiting at 2048,
  4 running at 2112). PP3 MTP2: `MAXBAT=2112` (2048 + 4 seqs × 3). TP2×PP2
  MTP2 at 8 seqs: `MAXBAT=4096`. Sub-block chunks are not an option (they
  poison the prefix cache, upstream #43559).
- **Decode cadence.** Each request decodes every `pp_size` steps, so per-request
  decode at c>1 drops; MTP pays off up to ~4 streams, MTP0 above.
- **Memory.** Vision does not fit next to stage 0's layers (text-only), dense
  INT8 must be `ONLY` mode, and the MTP drafter needs the W4A16 drafter
  checkpoint (`tools/rdna/quant_mtp_experts.py`). Under PP the drafter shares
  the last stage's target embedding and `lm_head` (CPU placeholders until
  shared). On 30 GB V620s a full 262K request at MTP2 needs `GMEM=0.975`
  (0.95 caps the length at ~188K); 32 GB cards fit it at 0.95.
- **Partition.** 16,16,16 for MTP0; 17,17,14 for MTP2 (the drafter sits on the
  last stage). PLE layers must stay on stage 0.

Measured on 3× / 4× V620 (int4 PLE, F&P, `--temperature 0`, 2026-10-13):

| Config | Cell | Prefill tok/s/req (agg) | Decode tok/s/req (agg out) | TTFT | ITL med |
|---|---|---|---|---|---|
| PP3 MTP0 (`flashnext-pp3`, KV 398K tok) | 1k/512 c=1 | 842 (835) | 38.4 (35.2) | 1.22 s | 26.1 ms |
| | 1k/512 c=8 | 204 (1078) | 9.0 (66.3) | 5.01 s | 106 ms |
| | 16k/1k c=1 | 1848 (1811) | 38.0 (28.6) | 8.87 s | 26.4 ms |
| | 16k/1k c=8 | 771 (1589) | 7.7 (53.3) | 21.3 s | 63.4 ms |
| PP3 MTP2 (`flashnext-pp3-mtp2`, KV 268K tok) | 1k/512 c=1 | 845 (842) | 57.7 (50.8) | 1.21 s | 36.3 ms |
| | 1k/512 c=4 | 535 (847) | 12.4 (46.1) | 1.92 s | 120 ms |
| | 16k/1k c=1 | 1866 (1839) | 70.9 (44.1) | 8.78 s | 37.0 ms |
| | 16k/1k c=4 | 1071 (3088) | 10.1 (34.7) | 15.3 s | 122 ms |
| TP2×PP2 MTP2 (`flashnext-mtp2 TP=2 PP=2 MAXBAT=4096`) | 1k/512 c=1 | 1496 (1453) | 52.6 (49.2) | 0.68 s | 37.8 ms |
| | 1k/512 c=8 | 565 (1309) | 13.7 (100.5) | 1.81 s | 141 ms |
| | 16k/1k c=1 | 2199 (2189) | 54.8 (39.2) | 7.45 s | 38.1 ms |
| | 16k/1k c=8 | 1055 (2399) | 8.6 (59.8) | 15.5 s | 132 ms |

On four cards TP=4 MTP2 decodes faster (16k c=1 76–87 tok/s, 16k c=8 ITL
~62 ms) while TP2×PP2 has higher aggregate prefill at 16k c=8 (2399 vs
1050–1350 tok/s). PP2's concurrency gate shows the known prompt-1 near-tie
flip. Raw runs: infrastructure `bench_results/2026-10-13_pp3-mtp2/` and
`2026-10-13_bundle-final-ab/` (TP4).

### PLE table (Flash-Next)

| Mode | Env | Host RAM peak | Notes |
|---|---|---:|---|
| `PLE=bf16` | (none) | ~160 GB | Upstream pinned BF16 table. |
| `PLE=int4` | `VLLM_PLE_QUANT_DIR=<ples_int4>` | ~90 GB | INT4 table in pinned RAM, dequantized in the lookup. |
| `PLE=int4-disk` | + `VLLM_PLE_CPU_OFFLOAD=1` | ~45 GB | CPU sidecar serves mmapped shards (page cache). Set `VLLM_PLE_OFFLOAD_READY_TIMEOUT=3600` (default 600 s) and `expandable_segments:False`. |

All three are validated with MTP0/MTP2 under agentic load at speed parity.
Tables come from `primitive-ai/Qwen3.8-Flash-Next-PLE-quant` (only `ples_int4`
is wired). `PLE_OFFLOAD_DOORBELL` (1) and `PLE_OFFLOAD_PREFAULT` (1) are
sidecar internals; leave them on.

### EXL3

`VLLM_EXL3_FOLDED_CACHE=<dir>` caches folded fp16 weights across loads.
`VLLM_EXL3_MEMORY_MODE` (`full`|`packed`), `VLLM_EXL3_M_MAX` (64),
`VLLM_EXL3_FUSED_PROJECT` (1), `VLLM_EXL3_PREFILL_DECODE` (1),
`VLLM_EXL3_GEMM_V3` (1), `VLLM_EXL3_V3_BLOCK_K` (0) and `VLLM_EXL3_HADAMARD`
(`both`) select kernel variants. `VLLM_EXL3_DEQUANT_ALL=1` forces the fp16
path (correctness A/B only).

### Other

- `VLLM_USE_BREAKABLE_CUDAGRAPH`: auto-enabled for models whose RDNA path
  needs eager breaks inside FULL graphs (DeepSeek-V4, Qwen3.5/3.8 hybrids).
  Set it explicitly only to override.
- `VLLM_HIPPIHX`, `_LIB`, `_CODE_OBJECT`: route ops through a prebuilt hippihx
  V1 library when its plan is ready.
- `VLLM_NO_PARTIAL_HASH=1`: disable partial-block prefix hits (debugging).

### Diagnostics (off by default; slow, do not bench with them)

NaN tracing: `VLLM_{LAYER,MOE,HC,GDN,PLE}_NAN_DEBUG`, `VLLM_RDNA2_MOE_DEBUG_NAN`.
Kernel launch logs: `VLLM_RDNA2_W4A8_DEBUG`, `VLLM_RDNA2_W4A8_MOE_DEBUG`,
`VLLM_RDNA2_PREFILL_DEBUG`, `VLLM_QSA_RDNA2_DEBUG`, `VLLM_CONV1D_DEBUG`,
`VLLM_EXL3_DEBUG`, `VLLM_EXL3_HADAMARD_DBG`, `VLLM_PAGED_MQA_DEBUG_OOB`.
Cache/scheduler traces: `VLLM_DEBUG_PREFIX_HIT`, `VLLM_TRACE_PREFIX_CACHE`.
Timing: `VLLM_RDNA_PHASE_TIMING=<every N steps>` (+ `VLLM_RDNA_OP_TIMING=1|2`),
`VLLM_RDNA_SYNC_DEBUG`. PLE sidecar:
`PLE_OFFLOAD_DEBUG_{HOPS,TRACE,DELAY_MS}`, `PLE_OFFLOAD_FUSED_CHECK`.
GPU profiling: the launchers set `ROCPROFILER_REGISTER_ENABLED=0`, so the HIP/HSA
runtimes never register with rocprofiler (no tool init, no queue interception)
while serving. Set `ROCPROFILER_REGISTER_ENABLED=1` for torch-profiler or
rocprofv3 GPU traces.

## TunableOp

gfx1030 GEMMs run on rocBLAS (hipBLASLt is off: `TORCH_BLAS_PREFER_HIPBLASLT=0`,
`PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED=0`). TunableOp replaces rocBLAS's heuristic
pick with a measured solver per exact GEMM shape `(op, M, N, K)`. Tuned rows
run ~15 % faster than the heuristic, and much faster on skinny shapes.

**What ships**: `tunableop/<profile>/tunableop_results{0..3}.csv` (one file per
rank) plus `tunableop/profiles.json`. Solver IDs only mean something to the
rocBLAS build that produced them, so one folder per build:
`rocm7.14-rocblas5.5` is the canonical serving profile and covers Flash-Next
MTP0/2 (including mixed-step and padded HC shapes), 27B AWQ (W4A8 0/1),
27B EXL3, Qwen3.6-35B-A3B GPTQ and DeepSeek-V4-Flash. Keep every row for a
build in that one folder; do not start parallel row sets.

**How it is consumed**: the launcher sources `tools/rdna2_028/tunableop_env.sh`,
which hashes the loaded `librocblas.so.5`. It then picks the profile whose
`lib_sha256[:12]` matches and sets `PYTORCH_TUNABLEOP_ENABLED=1`, `TUNING=0`
(lookup only) and `PYTORCH_TUNABLEOP_FILENAME` at those rows.
`TUNABLEOP_PROFILE=<name>` forces a profile and fails the launch on a hash
mismatch. If no profile matches, it falls back to
`$HOME/.cache/tunableop/tunableop_results.csv`. Look for
`TunableOp profile … auto-selected` in the log. **Never** point rows at `/tmp`
or the run CWD.

A shape without a row silently runs the heuristic. Shapes change when you
change `MAXBAT`, `BLOCK_SIZE` (chunk alignment), MTP, TP or the model. The
HC pad (`VLLM_RDNA_HC_PAD_M`) exists to keep mixed steps on tuned rows.

**Generating rows** (service stopped, GPUs exclusive, everything under a
persistent work dir):

1. **Census.** Serve the target workloads with
   `PYTORCH_TUNABLEOP_UNTUNED_FILENAME=<work>/untuned%d.csv` (lookup still on)
   to record shapes that have no row. Not `_RECORD_UNTUNED_*`; see
   `tools/rdna2_028/probe_record_untuned.py`.
2. **Tune.** `PHASE=tune tools/rdna2_028/tunableop_rows_pipeline.sh` (driver
   `tune_prod_shapes.py --mode tune`) tunes each shape on every rank into
   `<WORK>/scratch/`. It also adds the M ≤ 8 decode shapes of every (N, K)
   seen.
3. **Measure + curate.** `PHASE=measure` times each shape three ways (heuristic,
   current rows, new rows). `curate_tunableop_rows.py` keeps the fastest per
   shape, drops rows slower than the heuristic, and omits `Default` rows.
4. **Merge and prove.** Merge into `tunableop/<profile>/`, update
   `rows_per_rank` in `profiles.json`, and run `verify_tunableop_lookup.py
   --rows <file>` to show each shipped row is actually hit. Then rerun the
   serve A/B: tuning can change reduction order, so check correctness gates
   as well as speed.

For a new rocBLAS build (or gfx11), make a new profile folder with its own
`lib_sha256` and run the full census → tune → curate → merge campaign.
`tunableop/README.md` has the per-profile provenance details.
