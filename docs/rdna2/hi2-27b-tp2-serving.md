# HI2 on two V620: serving Qwen3.8-27B-HI2-3.0bpw at TP=2

HI2 is the user-facing name of the HadamardINT2-FP16 weight format (spec v3.2): each weight
row is stored as 2, 4 or 8 bit codes after a Hadamard transform, and the checkpoint
`BlivionIaG/Qwen3.8-27B-HI2-3.0bpw` averages 3.0 bits per weight. In the code and in the
checkpoint the name is unchanged: `quant_method: rdna_hint_flex` (alias `hadamard_int2`),
tensor prefix `hi2_`. The model is Qwen3.8-27B, a dense hybrid with 48 GDN (linear attention)
layers and 16 full-attention layers. Two V620 cards (gfx1030, 29.98 GiB each) serve it at
TP=2.

This page is the user guide: how to serve, what every switch does and costs, the limits, and
the measurements behind the recipe defaults. The format and the kernels are in
[`rdna-hint-flex-weights.md`](rdna-hint-flex-weights.md). Labels: **[measured]** = a
scoreboard arm listed below, **[derived]** = arithmetic on measured values, **[inference]** =
a run must confirm it.

## Serve

From the repository root. `MODEL` is the snapshot directory, `VENV` a ROCm 7.14 venv that has
this tree's extensions built (the measurements used `venv-7.14.0_0.32.0`).

```bash
bash tools/rdna/serve_rdna.sh RECIPE=27b-hi2-tp2 MODEL=/path/to/Qwen3.8-27B-HI2-3.0bpw \
  VENV=/path/to/venv HIP_VISIBLE_DEVICES=2,3
```

The server listens on `127.0.0.1:18211` as model `q27hi2` and is ready after 160 to 190 s
[measured: `READY after` in the arms below]. Add `HOST=0.0.0.0` to listen on the network.
`PRINT=1` prints the environment, the TunableOp profile and the exact command without
launching. A request:

```bash
curl -s http://127.0.0.1:18211/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"q27hi2","messages":[{"role":"user","content":"Say hello."}],"max_tokens":32}'
```

Two recipes, identical except `MTP` (a test keeps them so):

| You want | Recipe | MTP | Why |
| --- | --- | ---: | --- |
| Throughput, or eight or more concurrent requests | [`27b-hi2-tp2`](../../tools/rdna/recipes/27b-hi2-tp2.env) | 0 | MTP does not raise throughput at c=8 and adds an ITL tail (see Known limits) |
| One user, or up to about four concurrent requests | [`27b-hi2-tp2-mtp2`](../../tools/rdna/recipes/27b-hi2-tp2-mtp2.env) | 2 | c=1 decode 85.5 / 68.8 / 58.4 tok/s per request at 1k / 16k / 32k, against 45.2 / 34.9 / 28.3 |

Precedence, lowest to highest: recipe file, variables already in the environment, trailing
`KEY=value` arguments. Every switch below can therefore be turned off on the command line:

```bash
bash tools/rdna/serve_rdna.sh RECIPE=27b-hi2-tp2 MODEL=... VENV=... HIP_VISIBLE_DEVICES=2,3 \
  VLLM_RDNA_GDN_DECODE_INPLACE=0
```

What a good boot logs (each line is printed once by a worker or the launcher):

- `TunableOp profile rocm7.14-rocblas5.5-hi2-27b-tp2 preferred (TUNABLEOP_PREFER) for rocBLAS
  build f30bb442e9b5 ...` and `TunableOp rows per rank ...: 0=3366 1=3366`.
- `RDNA_ATTN: unfused prefill attention enabled (query rows >= 64, scratch <= 768 MiB, ...)`,
  and `unfused prefill attention is serving long chunks` after the first long prompt.
- `VLLM_RDNA_GEMMA_RESIDUAL_DIRECT=1: GemmaRMSNorm with a residual uses the cached fp16 scale
  ...`.
- `Model loading took 6.79 GiB` (rank 1: 6.80): 7.46 GiB means the GDN state arenas were
  allocated, so `VLLM_RDNA_GDN_DECODE_INPLACE` did not take effect. When it is refused the
  worker logs `VLLM_RDNA_GDN_DECODE_INPLACE=1 ignored, the GDN decode keeps its state arenas:
  <reason>`.
- `RDNA_HINT_FLEX_GEMV`, `RDNA_HINT_FLEX_FUSE_SILU`, `VLLM_RDNA_LM_HEAD_GEMV_MAX_M` and
  `RDNA_HINT_FLEX_PREFILL_GEMM` log nothing when they take effect; `PRINT=1` shows the values
  the launcher exports. A `_rocm_C` older than this tree makes the model log one warning
  (`this _rocm_C build has no rdna_hint_flex_dense_gemm_rdna2; prefill stays on rocBLAS`).
- `Unknown vLLM environment variable detected: VLLM_RDNA_...` warnings from the API server
  are cosmetic: the fork reads these variables directly and does not register them in
  `vllm/envs.py` (the allreduce and conv1d variables of every recipe do the same).

## Switches

Every switch is off in the code (no other recipe, model or backend sees any of them) and on
in `27b-hi2-tp2` and `27b-hi2-tp2-mtp2`. The first column is what the recipes set, the last
column the one-line opt-out. `tests/tools/rdna/test_hi2_recipe_defaults.py` reads this table
and fails when a recipe and the table disagree, or when an opt-out does not win over the
recipe (on the command line and in the caller's environment). Effects are scoreboard arms
against the flags-off `base` arm (the last row: against `final-a2`, which has the other
seven on), see [Measured results](#measured-results).

| Recipe default | What it does | Measured effect | Opt-out |
| --- | --- | --- | --- |
| `RDNA_HINT_FLEX_GEMV=auto` | Decode GEMV for 2 to 48 rows (c >= 2 and the MTP verify step) runs the v2 kernel, which reads each weight once for all rows; one row keeps v1. Not batch-invariant: a token decoded alone and in a batch can differ by one fp16 ulp in a logit (`v2` is invariant, about 3% slower at c=1). | arm `gemv`: 1k/512 c=8 92.5 to 117.8 tok/s (x1.27), decode 13.8 to 18.8 tok/s per request, ITL p99 63.4 to 45.1 ms; 16k/1k c=8 x1.05; c=1 unchanged | `RDNA_HINT_FLEX_GEMV=v1` |
| `VLLM_FA_RDNA2_UNFUSED_PREFILL=1` | Prefill attention (head_dim 256, fp16 KV) runs as two rocBLAS GEMMs and an in-place softmax per query block instead of the flash kernel. Decode, verify steps, chunks under 64 query rows and sequences whose scratch would exceed 768 MiB keep the flash kernel. fp16 scores. | arm `attn`: TTFT 16k/1k c=1 30.54 to 24.64 s, 32k/2k c=1 75.75 to 53.22 s, 16k/1k c=8 97.14 to 79.33 s; 1k cells and decode unchanged | `VLLM_FA_RDNA2_UNFUSED_PREFILL=0` |
| `VLLM_RDNA_GEMMA_RESIDUAL_DIRECT=1` | Gemma-style RMSNorm with a residual uses a cached fp16 scale and one in-place fused add-norm kernel (6 launches become 1, 128 calls per step). | arm `residual`: 1k/512 c=1 33.0 to 35.3 tok/s (x1.07), c=8 x1.02; decode 36.5 to 39.3 tok/s per request; outputs at the boot-to-boot floor | `VLLM_RDNA_GEMMA_RESIDUAL_DIRECT=0` |
| `VLLM_RDNA_GDN_DECODE_INPLACE=1` | GDN decode of a pure non-speculative batch runs on the paged conv and SSM state in place: no per-layer state arenas (13 or 16 launches per GDN layer become 3). The arenas are 0.65 GiB per rank at MTP=0 and about 1.8 GiB at MTP=2 [derived]. Needs `VLLM_CAUSAL_CONV1D_RDNA2_UPDATE=0` (the recipes set it); otherwise it logs a warning and keeps the arenas. | arm `gdnglue`: 1k/512 c=1 33.0 to 37.1 tok/s (x1.12), c=8 x1.08; decode 36.5 to 41.6 tok/s per request; outputs at the floor; `Model loading took` 7.46 to 6.79 GiB | `VLLM_RDNA_GDN_DECODE_INPLACE=0` |
| `RDNA_HINT_FLEX_FUSE_SILU=1` | The dense MLP's SiluAndMul is folded into the down projection's activation transform: 64 launches fewer per step, outputs bit-identical to the unfused path (unit tests). | Not isolated: arm `stack` holds all five switches. Microbenchmark 0.19 to 0.20 ms per step (0.7% of a c=1 step). `stack` against `residual` times `gdnglue` at c=1: x1.01 [derived] | `RDNA_HINT_FLEX_FUSE_SILU=0` (only `0` or empty turn it off: `false` counts as on) |
| `VLLM_RDNA_LM_HEAD_GEMV_MAX_M=32` | The LM head takes the dense GEMV for up to 32 rows (code default 8). It matters only when a step has more than 8 logit rows: MTP=2 with three or more requests (three rows each). At MTP=0 a step has at most `SEQS` = 8 rows and nothing changes. HIP 4 microbenchmark, 12 rows: rocBLAS 11.4 ms, GEMV 5.1 ms; 24 rows: 11.7 and 7.4 ms. | arms `mtp2-nofix` to `mtp2-lmfix` (MTP=2, no other switch): c=4 82.0 to 86.2 tok/s (x1.05), ITL p99 118.8 to 92.6 ms; c=8 98.6 to 100.4; c=1 and c=2 unchanged | `VLLM_RDNA_LM_HEAD_GEMV_MAX_M=8` |
| `TUNABLEOP_PREFER=rocm7.14-rocblas5.5-hi2-27b-tp2` | TunableOp rows for the rocBLAS prefill GEMMs: the canonical `rocm7.14-rocblas5.5` rows plus 36 rows per rank for this model (gate_up, in_proj_qkvz, qkv and out / o_proj at M = 1023, 1024, 2046 to 2048; in_proj_ba at the census M). A soft preference: used only when the profile is registered for the loaded `librocblas.so.5` (sha256 `f30bb442e9b5...`), otherwise the launcher warns and loads the canonical rows. The rows are build-specific. | arm `tunable`: TTFT 1k/512 c=1 1.51 to 1.41 s, 16k/1k c=1 30.54 to 28.68 s, 32k/2k c=1 75.75 to 72.09 s; output x1.01 to x1.03 at c=1, within noise at c=8; concurrency gate 0 of 480 checks failing with and without the rows | `TUNABLEOP_PREFER=` (empty); `TUNABLEOP_PROFILE=<name>` wins over it |
| `RDNA_HINT_FLEX_PREFILL_GEMM=hip` | Dense HIP prefill GEMM above M = 48 in place of dequant + rocBLAS, per shape where its dispatch table measured faster (rocBLAS everywhere else). It rounds differently from rocBLAS (see Quality). | arm `final2-a2` against `final-a2`, same card pair: 16k/1k c=8 30.3 to 34.8 tok/s (x1.15), TTFT 75.95 to 64.08 s; 1k/512 c=8 137.7 to 147.7 (x1.07); TTFT 16k/1k c=1 21.81 to 20.45 s, 32k/2k c=1 47.43 to 44.77 s; c=1 decode unchanged. Microbenchmark 1.09 to 1.12x over tuned rocBLAS at M = 2048 | `RDNA_HINT_FLEX_PREFILL_GEMM=rocblas` |

Notes on the table:

- `TUNABLEOP_PREFER` is a launcher variable: it is not exported to the server, so `PRINT=1`
  shows its effect in the `# tunableop:` line and not in the environment block.
- The switches are read when the process or the model is built: set them before the launch.
- `RDNA_HINT_FLEX_PREFILL_GEMM` accepts `""`, `0`, `off` and `rocblas` for rocBLAS, and `hip`
  or `hip-notail` for the dense kernel; any other value raises an error at start.
- `VLLM_RDNA_GEMMA_RESIDUAL_DIRECT` is global for every `GemmaRMSNorm` with a residual in the
  process and is guarded only for the Qwen3.5 / Qwen3.8 decoders: do not carry it over to
  gemma, gemma2, gemma3, rnj1 or interns2_mobius
  ([`hi2-residual-direct-2026-11-15.md`](hi2-residual-direct-2026-11-15.md)).
- Values. Only the exact value `1` turns on `VLLM_FA_RDNA2_UNFUSED_PREFILL`,
  `VLLM_RDNA_GEMMA_RESIDUAL_DIRECT` and `VLLM_RDNA_GDN_DECODE_INPLACE` (`true` is off).
  `RDNA_HINT_FLEX_FUSE_SILU` is off only for empty or `0`. `RDNA_HINT_FLEX_GEMV` takes `v1`,
  `v2` or `auto`, case-sensitive, and any other string means `v1`. `VLLM_RDNA_LM_HEAD_GEMV_MAX_M`
  must be an integer (clamped to 8 to 32): `32.0` raises when vLLM imports `layers/utils.py`.
- Compile cache. `RDNA_HINT_FLEX_FUSE_SILU` and `RDNA_HINT_FLEX_PREFILL_GEMM` change the traced
  graph but are not part of the compile-cache key. The recipes turn the cache off
  (`VLLM_DISABLE_COMPILE_CACHE=1`, `VLLM_USE_AOT_COMPILE=0`), so toggling them between launches is
  safe there. If you turn the cache back on, give each setting its own `VLLM_CACHE_ROOT`
  [inference: the AOT load path was not traced].

## Slots and memory

**GPU slots.** TP=2 splits every weight in two. Pick the pair by PCI bus, not by HIP index
(an index is the enumeration order and moves when cards, risers or switches change):
`PCI_BUSES="d9 dc"` in rank order, which the launcher resolves to `HIP_VISIBLE_DEVICES` at
launch and refuses when a bus is missing or the count is not TP. The measured pair is HIP 2,3
= PCI buses `d9 dc` on `par1-cs25` (every arm below; `scoreboard.sh` refuses to boot when the
two do not match). Keep a tensor-parallel group on one PCIe switch when the host has several
([`docs/rdna/serving.md`](../rdna/serving.md)). The one-shot all-reduce
(`VLLM_RDNA_AR_MAX_KB=256`, `VLLM_RDNA_AR_ONESHOT_KB=256`) needs PCIe peer-to-peer between
the two cards: a decode step reduces 10 KiB per token (80 KiB at c=8, 240 KiB for a c=8 MTP=2
verify step), prefill chunks of 5 to 20 MiB stay on RCCL. On a host with broken P2P set
`P2P=0` (RCCL over host shared memory, one-shot off); no end-to-end number exists for that
setting. A worker reads `tunableop_results<r>.csv` for the device ordinal `r` it sees, so the
rows follow the rank, not the physical slot.

**Sequence slots and KV.** `SEQS=8` sequences, `MAXBAT=2048` (derived: two 1024-token mamba
state blocks), `BLOCK_SIZE=1024`, `MAXLEN=262144`. `KV=17179869184` (16 GiB per rank, fixed:
it skips the memory profile). One token costs 32 KiB of KV per rank (16 full-attention layers
x 2 KV heads x 256 x K and V x 2 B), so the pool holds 512,281 tokens at MTP=0 (1.95 times
one 262,144-token request) and 470,489 at MTP=2 (1.79 times) [measured: boot log]. Eight
requests of 32k tokens (262,144) fit; eight of 64k do not [derived].

| Per card | `base` (switches off) | `final` (MTP=0) | `final-mtp2` |
| --- | ---: | ---: | ---: |
| `Model loading took` (GiB) | 7.46 / 7.47 | 6.79 / 6.80 | 7.19 / 7.19 |
| KV pool (tokens) | 512,281 | 512,281 | 470,489 |
| VRAM max over the arm (MiB, sysfs, 1 s samples) | 27,143 / 27,151 | 27,022 / 27,030 | 27,595 / 27,601 |
| Free at that maximum (of 30,704 MiB) | 3.5 GiB | 3.6 GiB | 3.0 GiB |

Where the weight memory goes. The recipes' CPU accounting of the real TP=2 load is 6.68 GiB
per rank (hi2 codes 3.80, block parameters 0.44, small fields 0.04, `lm_head` 1.18,
`embed_tokens` 1.18, rope cache 0.03), but the engine logged 7.46 GiB. The GDN state arenas
explain most of the 0.78 GiB. They are allocated at load, one per GDN layer, with
`max_num_seqs x (num_spec + 1) + 1` rows of 1.53 MiB each: 0.65 GiB per rank at MTP=0 and
about 1.8 GiB at MTP=2 [derived]. `VLLM_RDNA_GDN_DECODE_INPLACE=1` does not allocate them:
7.46 to 6.79 GiB at MTP=0 and 9.03 to 7.19 GiB at MTP=2 [measured: arms `base`, `gdnglue`,
`mtp2-lmfix`, `final-mtp2`]. With the arenas gone the MTP drafter costs 0.40 GiB (7.19
against 6.79), which is its one layer (0.395 GiB per rank): no second copy of `embed_tokens`
or `lm_head` stays resident for it, and the guess to the contrary in
`hi2-27b-tp2-2026-11-14.md` is wrong. About 0.11 GiB (6.79 against 6.68) stays unexplained.

## Known limits

- **MTP at c=8 gives no throughput and a long ITL tail** [measured]. 1k/512 c=8: 134.0
  against 135.4 tok/s, ITL p50 94.1 against 35.2 ms, ITL p99 1,624.5 against 39.6 ms; 16k/1k
  c=8: 28.6 against 29.4 tok/s. The cause was not investigated. [inference] The decode-stall
  cap is off in these recipes (`VLLM_RDNA_DECODE_STALL_MS=0`, see
  [`mixed-batch-stall-probe.md`](mixed-batch-stall-probe.md)), a 1k prompt chunk takes about
  1.4 s at 717 tok/s, and speculative requests finish at different steps, so a new request's
  prefill chunk lands in a decode step of the others. At 16k/1k c=8 every arm, MTP=0 included,
  has an ITL p99 of 1.8 to 2.5 s. Use `27b-hi2-tp2` from c=8.
- **The MTP numbers are for random prompts.** The bench cells use random-token prompts with
  greedy decoding. 85.5 tok/s per request at 1k with a 33.4 ms median step is about 2.85
  tokens per step [derived], close to the maximum of 3. The acceptance probe on 24 natural-text
  prompts measured 2.31 tokens per step at c=1 and c=4 (draft acceptance 65%, per position
  0.76 and 0.54). [inference] At that rate and the same step time a request decodes at about 69
  tok/s at 1k, 1.5 times MTP=0 and not the bench's 1.9 times. The probe's own end-to-end rate
  was 63.6 tok/s at c=1 and 123.6 tok/s aggregate at c=4 (24 requests of about 254 tokens each,
  prefill included, "not a benchmark"). MTP=0 at c=2 and c=4 was not measured: the break-even
  at c=4 is [inference] (an earlier kernel-time estimate on the old stack gave about 1.5x at
  c=1, 1.4x at c=2 and 1.0x at c=4, `hi2-27b-tp2-2026-11-14.md`).
- **`lm_head` in int8 is not validated.** `VLLM_RDNA_DENSE_INT8=1` halves the head's time (1.46
  against 2.91 ms at one row, 1.45 ms per step, 5% of a c=1 step [measured: microbenchmark]),
  but per-row int8 of a vocabulary-sized fp16 matrix is lossy and no `decode_kl.py` run against
  the fp16 head exists. No recipe enables it. Do not, before such a run.
- **0.11 GiB of the weight memory is unexplained** (see Slots and memory).
- **Not bit-identical to the old defaults.** Long-context PPL 3.6633 to 3.6635 (3.6631 with the
  dense prefill GEMM), prompt KL 7.7e-6 nats per token (7.8e-6; top-1 agreement 0.9986 to
  0.9987), decode KL 7.0e-6 to 8.1e-6 against a boot-to-boot floor of 2.4e-6, no confident
  flip; the scoreboard judges these "above floor" because the floor of the prompt KL is
  exactly 0. `residual` and `gdnglue` alone sit at the floor; the KL comes from `attn`, the
  tuned rows, `gemv` and the dense GEMM (see Quality). `RDNA_HINT_FLEX_GEMV=auto` is not
  batch-invariant, see the table.
- **The TunableOp rows are build-specific.** They are registered for `librocblas.so.5` sha256
  `f30bb442e9b5...` (rocBLAS 5.5.0.cd957402, ROCm 7.14, torch 2.12.0+rocm7.14.0). Another build
  falls back to the canonical rows with a warning; solver ids do not carry over, so re-tune with
  `tools/rdna/rdna_hint_flex/tunableop_hi2.sh` on a new build.
- **The dense prefill GEMM rounds differently from rocBLAS** [measured]. With it, long PPL is
  3.6631 (3.6635 without), prompt KL 7.80e-6 (7.73e-6) against `base`, decode KL 8.1e-6
  (8.0e-6 without, same card pair) and there is still no confident flip, but a prompt
  prefilled by it can differ from one prefilled by rocBLAS in a late token. It is part of the
  gfx10x kernel set of `rdna_hint_flex`; an older `_rocm_C` keeps rocBLAS with one warning.
  Opt out with `RDNA_HINT_FLEX_PREFILL_GEMM=rocblas`.
- **Scope.** Measured on two V620 (gfx1030) at TP=2 only. TP=1, TP=4 and gfx11 were not run
  with these recipes. The 262,144-token context fits the KV pool, but no benchmark cell goes
  beyond 32k.

## Scoreboard protocol

Every number below comes from the scoreboard ([`hi2-scoreboard-2026-11-14.md`](hi2-scoreboard-2026-11-14.md)
has the full description): one **arm** is one boot of a recipe with a set of overrides, run by
`tools/rdna/rdna_hint_flex/scoreboard.sh ARM [KEY=value ...]`. Per arm:

- five bench cells (`input:output:concurrency:requests`): `1024:512:1:16`, `1024:512:8:16`,
  `16384:1024:1:8`, `16384:1024:8:16`, `32768:2048:1:4`; `vllm bench serve` with random-token
  prompts, seed 12345, `--ignore-eos`, `--temperature 0`, request rate inf, the prefix cache
  cleared before every cell;
- quality on the live server: short PPL, long-context PPL (six passages of 15k to 23k tokens
  and 96 greedy tokens each), prompt-level KL / dNLL / top-1 (64 windows of 2048 tokens),
  decode-path KL (32 passages x 512 greedy tokens at 8 concurrent);
- validity: greedy probe before and after load, PPL probe, prefix-cache probe, deterministic
  concurrency probe (48 checks), KFD queue evictions per cell (a non-zero cell is INVALID),
  amdgpu fault markers, junction temperature (limit 90 C) and VRAM;
- the quiet protocol (`QUIET_HOLD=1`): the driver waits for any `docker buildx build` on the
  box and for other lanes' windows, so no cell shares the GPUs or the host with a build.
  An arm without it is labelled "not for comparison".

`BASELINE=` and `FLOOR=` name earlier arms: quality is compared with the baseline, and the
floor (the baseline booted a second time) is the boot-to-boot noise a delta must exceed.
Boot-to-boot output spread on `base` against `base2` is at most 0.1% in every cell.

The recipes now turn the switches on, so `scoreboard.sh base` boots with them. The flags-off
baseline of the tables below names the opt-outs:

```bash
OFF="RDNA_HINT_FLEX_GEMV=v1 VLLM_FA_RDNA2_UNFUSED_PREFILL=0 VLLM_RDNA_GEMMA_RESIDUAL_DIRECT=0
 VLLM_RDNA_GDN_DECODE_INPLACE=0 RDNA_HINT_FLEX_FUSE_SILU=0 VLLM_RDNA_LM_HEAD_GEMV_MAX_M=8
 TUNABLEOP_PREFER="
bash tools/rdna/rdna_hint_flex/scoreboard.sh base  $OFF            # the baseline
bash tools/rdna/rdna_hint_flex/scoreboard.sh base2 $OFF            # its floor
BASELINE=base FLOOR=base2 bash tools/rdna/rdna_hint_flex/scoreboard.sh ship              # MTP=0 recipe defaults
BASELINE=base FLOOR=base2 RECIPE=27b-hi2-tp2-mtp2 \
  bash tools/rdna/rdna_hint_flex/scoreboard.sh ship-mtp2                                  # MTP=2 recipe defaults
python tools/rdna/rdna_hint_flex/scoreboard_report.py table \
  ~/w4a8_runs/scoreboard/{base,ship,ship-mtp2}                                            # the tables
```

The scoreboard defaults to HIP 2,3 = PCI `d9 dc` (`GPUS`, `BUSES`), `PORT=18211`,
`OUT_ROOT=$HOME/w4a8_runs/scoreboard`, `VENV=$HOME/Apps/vllm/venv-7.14.0_0.32.0`; set them
for another host. `scoreboard_report.py` needs the `regex` module (the venv has it); the
top-k comparison needs numpy. A single boot with the probes and cells, without the quality
passes: [`serve_validate.sh`](../../tools/rdna/port_v031/serve_validate.sh):

```bash
RECIPE=27b-hi2-tp2 MODEL=/path/to/Qwen3.8-27B-HI2-3.0bpw GPUS=2,3 PORT=18211 TAG=hi2 \
  CELLS="1024:512:1 1024:512:8 16384:1024:1" BENCH_EXTRA="--temperature 0" \
  bash tools/rdna/port_v031/serve_validate.sh
```

## Measured results

All arms: quiet protocol, KFD evictions 0 in every window, junction temperature at most 86 C,
greedy / PPL / prefix-cache / concurrency probes PASS. Arms without a suffix ran on HIP 2,3
(PCI `d9 dc`) of `par1-cs25`, arms with the suffix `-a2` on HIP 0,1 (PCI `d1 d4`). The second
pair reads 1 to 3% faster, so compare arms only within one pair. Raw files:
`bench_results/2026-11-15_hi2-scoreboard/<arm>/` in this repository (`meta.json` and
`scoreboard.md` of every arm named below, `scoreboard.json` of `base`, `final`, `final-mtp2`,
`gemm` and the five `-a2` arms, `runs.tsv` of the two concurrency arms) and
`~/w4a8_runs/scoreboard/<arm>/` on the box, which has every file of every arm. The first
table is `scoreboard_report.py table` output with columns reduced (commands below); the other
tables are read from the per-arm `scoreboard.md`.

```bash
python tools/rdna/rdna_hint_flex/scoreboard_report.py table \
  bench_results/2026-11-15_hi2-scoreboard/{base,final,final-mtp2}
python tools/rdna/rdna_hint_flex/scoreboard_report.py table \
  bench_results/2026-11-15_hi2-scoreboard/{final-a2,ship-a2,final2-a2}
```

### Recipe defaults against flags off

`base` = every switch off, canonical TunableOp rows. `final` = the switches of `27b-hi2-tp2`,
MTP=0. `final-mtp2` = `final` with `MTP=2`. Tokens per second are output tokens per second of
the cell; the decode column is per request; TTFT is the mean over the cell's requests.

| Cell | Arm | Output tok/s | TTFT (s) | Prefill tok/s | Decode tok/s per request | ITL p50 / p99 (ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1k/512 c=1 | `base` | 33.0 | 1.51 | 678 | 36.5 | 27.4 / 30.5 |
| 1k/512 c=1 | `final` | 40.3 | 1.39 | 737 | 45.2 | 22.1 / 25.0 |
| 1k/512 c=1 | `final-mtp2` | 68.6 | 1.43 | 717 | 85.5 | 33.4 / 33.9 |
| 1k/512 c=8 | `base` | 92.5 | 7.00 | 172 | 13.8 | 62.0 / 63.4 |
| 1k/512 c=8 | `final` | 135.4 | 6.97 | 209 | 22.4 | 35.2 / 39.6 |
| 1k/512 c=8 | `final-mtp2` | 134.0 | 7.41 | 204 | 24.9 | 94.1 / 1624.5 |
| 16k/1k c=1 | `base` | 15.7 | 30.54 | 536 | 29.6 | 33.8 / 34.2 |
| 16k/1k c=1 | `final` | 19.7 | 22.75 | 720 | 34.9 | 28.6 / 31.8 |
| 16k/1k c=1 | `final-mtp2` | 26.9 | 23.14 | 708 | 68.8 | 41.9 / 43.5 |
| 16k/1k c=8 | `base` | 23.2 | 97.14 | 264 | 4.6 | 90.1 / 2456.4 |
| 16k/1k c=8 | `final` | 29.4 | 78.63 | 322 | 6.1 | 63.5 / 1808.4 |
| 16k/1k c=8 | `final-mtp2` | 28.6 | 81.06 | 327 | 6.1 | 197.1 / 1920.5 |
| 32k/2k c=1 | `base` | 12.9 | 75.75 | 433 | 24.7 | 40.5 / 41.1 |
| 32k/2k c=1 | `final` | 16.8 | 49.40 | 663 | 28.3 | 35.4 / 37.5 |
| 32k/2k c=1 | `final-mtp2` | 23.9 | 50.48 | 649 | 58.4 | 50.6 / 51.8 |

Requests per cell: 16, 16, 8, 16, 4. The 16k/1k c=8 cell queues 16 requests behind 8 slots,
which is why its TTFT mean and ITL p99 are large in every arm.

### Effect of each switch

Output tok/s per cell; the last column is the TTFT of the 16k/1k c=1 cell. `stack` = the five
switches without rows, `final` = `stack` and the rows. `base2` is `base` booted again.

| Arm | Overrides over the flags-off recipe | 1k c=1 | 1k c=8 | 16k c=1 | 16k c=8 | 32k c=1 | TTFT 16k c=1 (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `base` | none | 33.0 | 92.5 | 15.7 | 23.2 | 12.9 | 30.54 |
| `base2` | none (floor) | 33.0 | 92.5 | 15.7 | 23.1 | 12.9 | 30.64 |
| `gemv` | `RDNA_HINT_FLEX_GEMV=auto` | 33.0 | 117.8 | 15.7 | 24.3 | 12.9 | 30.55 |
| `attn` | `VLLM_FA_RDNA2_UNFUSED_PREFILL=1` | 33.0 | 92.8 | 17.3 | 26.8 | 15.0 | 24.64 |
| `residual` | `VLLM_RDNA_GEMMA_RESIDUAL_DIRECT=1` | 35.3 | 94.4 | 16.2 | 23.3 | 13.2 | 30.50 |
| `gdnglue` | `VLLM_RDNA_GDN_DECODE_INPLACE=1` | 37.1 | 99.8 | 16.6 | 23.6 | 13.5 | 30.54 |
| `stack` | the five switches | 40.0 | 134.6 | 19.0 | 29.3 | 16.3 | 24.52 |
| `tunable` | `TUNABLEOP_PREFER=...hi2-27b-tp2` | 33.3 | 93.1 | 16.2 | 23.2 | 13.2 | 28.68 |
| `final` | the five switches and the rows | 40.3 | 135.4 | 19.7 | 29.4 | 16.8 | 22.75 |

### Dense prefill GEMM and the shipped recipes (card pair a2)

These arms ran on the other card pair (HIP 0,1, PCI `d1 d4`), so they compare only with each
other. `final-a2` is `final` again (the seven switches, no dense GEMM; tree `b47eb6c384`).
`ship-a2` is the recipe `27b-hi2-tp2` of `15a4bc6f49` with no environment variable at all
(before the dense GEMM was added to it): it reproduces `final-a2` within 0.1% on every cell,
with equal PPL and prompt KL. `final2-a2` is `final-a2` plus `RDNA_HINT_FLEX_PREFILL_GEMM=hip`,
which is the recipe as it is now.

| Cell | Arm | Output tok/s | TTFT (s) | Prefill tok/s | Decode tok/s per request | ITL p50 / p99 (ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1k/512 c=1 | `final-a2` | 40.7 | 1.35 | 757 | 45.5 | 22.0 / 22.6 |
| 1k/512 c=1 | `ship-a2` | 40.7 | 1.35 | 757 | 45.5 | 22.0 / 22.3 |
| 1k/512 c=1 | `final2-a2` | 40.9 | 1.27 | 806 | 45.5 | 22.0 / 22.4 |
| 1k/512 c=8 | `final-a2` | 137.7 | 6.76 | 215 | 22.7 | 34.9 / 36.4 |
| 1k/512 c=8 | `ship-a2` | 137.8 | 6.77 | 215 | 22.7 | 34.9 / 36.1 |
| 1k/512 c=8 | `final2-a2` | 147.7 | 5.72 | 250 | 23.5 | 34.9 / 36.2 |
| 16k/1k c=1 | `final-a2` | 20.1 | 21.81 | 751 | 35.2 | 28.5 / 28.9 |
| 16k/1k c=1 | `ship-a2` | 20.1 | 21.78 | 752 | 35.2 | 28.4 / 28.7 |
| 16k/1k c=1 | `final2-a2` | 20.7 | 20.45 | 801 | 35.2 | 28.4 / 28.8 |
| 16k/1k c=8 | `final-a2` | 30.3 | 75.95 | 333 | 6.2 | 63.0 / 1743.9 |
| 16k/1k c=8 | `ship-a2` | 30.3 | 75.88 | 334 | 6.2 | 62.8 / 1741.0 |
| 16k/1k c=8 | `final2-a2` | 34.8 | 64.08 | 394 | 6.9 | 62.9 / 1455.0 |
| 32k/2k c=1 | `final-a2` | 17.2 | 47.43 | 691 | 28.5 | 35.2 / 35.8 |
| 32k/2k c=1 | `ship-a2` | 17.2 | 47.44 | 691 | 28.5 | 35.1 / 35.5 |
| 32k/2k c=1 | `final2-a2` | 17.6 | 44.77 | 732 | 28.5 | 35.1 / 35.6 |

With MTP=2: `ship-mtp2-a2` is the recipe `27b-hi2-tp2-mtp2` of `15a4bc6f49`, `final2-mtp2-a2` is
`final2-a2` with `MTP=2`.

| Cell | Arm | Output tok/s | TTFT (s) | Prefill tok/s | Decode tok/s per request | ITL p50 / p99 (ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1k/512 c=1 | `ship-mtp2-a2` | 69.0 | 1.39 | 735 | 85.7 | 33.3 / 33.7 |
| 1k/512 c=1 | `final2-mtp2-a2` | 70.2 | 1.30 | 787 | 86.2 | 33.3 / 33.7 |
| 1k/512 c=8 | `ship-mtp2-a2` | 136.9 | 7.12 | 211 | 25.4 | 92.7 / 1581.2 |
| 1k/512 c=8 | `final2-mtp2-a2` | 146.6 | 5.51 | 276 | 25.6 | 92.6 / 1315.1 |
| 16k/1k c=1 | `ship-mtp2-a2` | 27.6 | 22.20 | 738 | 69.1 | 41.6 / 42.1 |
| 16k/1k c=1 | `final2-mtp2-a2` | 28.7 | 20.88 | 785 | 69.5 | 41.6 / 42.2 |
| 16k/1k c=8 | `ship-mtp2-a2` | 29.5 | 76.48 | 354 | 6.2 | 195.9 / 1855.3 |
| 16k/1k c=8 | `final2-mtp2-a2` | 32.9 | 68.12 | 374 | 6.8 | 196.0 / 1641.3 |
| 32k/2k c=1 | `ship-mtp2-a2` | 24.6 | 48.49 | 676 | 58.8 | 50.2 / 51.5 |
| 32k/2k c=1 | `final2-mtp2-a2` | 25.4 | 45.91 | 714 | 58.7 | 50.3 / 51.6 |

The effect of the dense GEMM on one card pair (output tok/s, TTFT mean; the ratio comes from
the unrounded values):

| Cell | MTP 0: `final-a2` to `final2-a2` | MTP 2: `ship-mtp2-a2` to `final2-mtp2-a2` |
| --- | --- | --- |
| 1k/512 c=1 | 40.7 to 40.9 (x1.01), 1.35 to 1.27 s | 69.0 to 70.2 (x1.02), 1.39 to 1.30 s |
| 1k/512 c=8 | 137.7 to 147.7 (x1.07), 6.76 to 5.72 s | 136.9 to 146.6 (x1.07), 7.12 to 5.51 s |
| 16k/1k c=1 | 20.1 to 20.7 (x1.03), 21.81 to 20.45 s | 27.6 to 28.7 (x1.04), 22.20 to 20.88 s |
| 16k/1k c=8 | 30.3 to 34.8 (x1.15), 75.95 to 64.08 s | 29.5 to 32.9 (x1.12), 76.48 to 68.12 s |
| 32k/2k c=1 | 17.2 to 17.6 (x1.02), 47.43 to 44.77 s | 24.6 to 25.4 (x1.03), 48.49 to 45.91 s |

The kernel runs only in prefill chunks of more than 48 rows, so the decode rate at c=1 does not
move (45.5, 35.2 and 28.5 tok/s per request at 1k, 16k and 32k). At 16k/1k c=8 the ITL p99
falls from 1743.9 to 1455.0 ms and the decode rate per request rises from 6.2 to 6.9 tok/s.
On the first pair, `gemm` (the switch and the tuned rows) against `tunable` (the rows alone)
points the same way:

| Cell | `tunable` | `gemm` |
| --- | ---: | ---: |
| 1k/512 c=1, tok/s | 33.3 | 33.4 |
| 1k/512 c=8, tok/s | 93.1 | 97.5 |
| 16k/1k c=1, tok/s | 16.2 | 16.7 |
| 16k/1k c=8, tok/s | 23.2 | 26.0 |
| 32k/2k c=1, tok/s | 13.2 | 13.5 |
| TTFT 16k/1k c=1 (s) | 28.68 | 26.87 |

### Quality

Short PPL is 5.730 in every arm. The reference is `base`, the floor `base2` (two boots of the
same configuration). Prompt KL is in nats per token; the decode columns come from 32
passages of 512 greedy tokens at 8 concurrent.

| Arm | Long PPL | Prompt KL | Top-1 agreement | Decode KL (mean) | Decode greedy identical | Long greedy identical |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `base` | 3.6633 | n/a | n/a | n/a | n/a | n/a |
| `base2` (floor) | 3.6633 | 0 | 1.0000 | 2.37e-6 | 28/32 | 6/6 |
| `gemv` | 3.6633 | 0 | 1.0000 | 6.18e-6 | 19/32 | 6/6 |
| `attn` | 3.6635 | 7.55e-6 | 0.9985 | 6.97e-6 | 19/32 | 6/6 |
| `residual` | 3.6633 | 0 | 1.0000 | 1.98e-6 | 28/32 | 5/6 |
| `gdnglue` | 3.6633 | 0 | 1.0000 | 2.40e-6 | 30/32 | 6/6 |
| `stack` | 3.6633 | 7.76e-6 | 0.9986 | 8.32e-6 | 16/32 | 5/6 |
| `tunable` | 3.6636 | 7.33e-6 | 0.9986 | 6.44e-6 | 22/32 | 6/6 |
| `final` | 3.6635 | 7.73e-6 | 0.9986 | 7.01e-6 | 21/32 | 5/6 |
| `final-mtp2` | 3.6635 | 7.73e-6 | 0.9986 | 8.36e-6 | 13/32 | 6/6 |
| `gemm` (HIP 2,3) | 3.6631 | 7.39e-6 | 0.9986 | 9.48e-6 | 20/32 | 6/6 |
| `final-a2` | 3.6635 | 7.73e-6 | 0.9986 | 8.04e-6 | 21/32 | 6/6 |
| `ship-a2` | 3.6635 | 7.73e-6 | 0.9986 | 7.14e-6 | 20/32 | 6/6 |
| `final2-a2` | 3.6631 | 7.80e-6 | 0.9987 | 8.13e-6 | 21/32 | 6/6 |
| `ship-mtp2-a2` | 3.6635 | 7.73e-6 | 0.9986 | 7.86e-6 | 18/32 | 6/6 |
| `final2-mtp2-a2` | 3.6631 | 7.80e-6 | 0.9987 | 8.08e-6 | 21/32 | 5/6 |

Zero confident flips in every decode-KL pass. `final`: dNLL +0.00001 (95% interval -0.00002 to
+0.00003), 5 of 6 long passages greedy-identical (first divergence at token 87.5 on average
against 96 for all six). The prompt KL is 0 for `gemv`, `residual` and `gdnglue` and about
7.5e-6 for `attn` and `tunable` each: a different attention path and a different GEMM solver
round differently (57 of the 89 candidate rows change a rounding, `hi2-27b-tunableop-2026-11-15.md`).
The decode KL of `gemv` is `auto`'s M=1 against M >= 2 difference (table above).

The dense prefill GEMM (`gemm`, `final2*`) moves the long PPL by -0.01% (3.6635 to 3.6631
against `final`) and the prompt KL by +0.07e-6; `gemm` has 7.39e-6 against `base`, the order of
the other switches. Quality does not depend on the card pair: `final-a2` has `final`'s PPL,
prompt KL and top-1. The decode KL and the identical count vary from boot to boot of one
configuration (`final` 7.0e-6 and 21/32, `final-a2` 8.0e-6 and 21/32, `ship-a2` 7.1e-6 and
20/32; floor 2.4e-6 and 28/32). There is still no confident flip.

### MTP arms

`final-mtp2` is in the first table, `ship-mtp2-a2` and `final2-mtp2-a2` in the dense GEMM
section. Acceptance (`spec_accept.py`, 24 natural-text prompts of about 254 tokens): c=1 mean
acceptance length 2.307, draft acceptance 65.4%, per position 0.763 and 0.544; c=4 2.315,
65.7%, 0.768 and 0.547; both PASS (`final2-mtp2-a2`, with the dense GEMM, measures the same
values). The two arms that isolate the LM-head cap run MTP=2 without the other switches:

| Cell | `mtp2-nofix` (cap 8) | `mtp2-lmfix` (cap 32) |
| --- | --- | --- |
| 1k/512 c=1 | 55.3 tok/s, decode 67.4 per request, ITL p99 42.6 ms | 55.2 tok/s, 67.4, 42.8 ms |
| 1k/512 c=2 | 75.2 tok/s, decode 49.1, ITL p99 55.9 ms | 75.2 tok/s, 49.1, 55.7 ms |
| 1k/512 c=4 | 82.0 tok/s, decode 27.2, ITL p99 118.8 ms | 86.2 tok/s, 30.2, 92.6 ms |
| 1k/512 c=8 | 98.6 tok/s, decode 16.6, ITL p99 1,659.7 ms | 100.4 tok/s, 16.6, 1,658.2 ms |

### Concurrency gate for the TunableOp rows

Arms `conc-on` (the rows) and `conc-off` (`TUNABLEOP_PREFER=`), both with
`VLLM_RDNA_AR_HIER=0`, no bench cells, `tools/rdna/rdna_hint_flex/conc_repeat_hook.sh`: ten
runs with `--ignore-eos` and six with `--no-ignore-eos`, 48 deterministic checks per run. Rows
on: 0 of 480 checks failing (471 identical, 9 near-ties) and 0 of 288 failing in the
`--no-ignore-eos` runs (287 identical, 1 near-tie). Rows off: 0 of 480 (471 identical, 9
near-ties) and 0 of 288 (all identical). Per-run lines: `conc-on/runs.tsv`, `conc-off/runs.tsv`.

### Arms, trees and result paths

Each tree sha is a commit in the history of this branch (`git checkout <sha>` reproduces the
arm). Overrides are over the recipe `27b-hi2-tp2` as that tree had it: up to `b47eb6c384` no
HI2 switch was on in it, at `15a4bc6f49` the first seven switches of the table are. From
`gemv` on it already carried `VLLM_RDNA_LM_HEAD_GEMV_MAX_M=32` (no effect at MTP=0, where a
step has at most 8 logit rows; `base`, `base2` and `attn` ran with 8), and from `tunable` on
`TUNABLEOP_PREFER=rocm7.14-rocblas5.5-hi2-27b-tp2` (`conc-on` relies on it; the other arms of
that tree name it, and `conc-off` empties it).

| Arm | Tree | Overrides |
| --- | --- | --- |
| `base`, `base2` | `ef0b99ea9b` (+ scoreboard tools) | none: the switches did not exist or were off |
| `gemv` | `f7edb8b1e7` | `RDNA_HINT_FLEX_GEMV=auto` |
| `attn` | `ef0b99ea9b` (+ scoreboard tools) | `VLLM_FA_RDNA2_UNFUSED_PREFILL=1` |
| `residual` | `dd1d104148` | `VLLM_RDNA_GEMMA_RESIDUAL_DIRECT=1` |
| `gdnglue` | `dd1d104148` | `VLLM_RDNA_GDN_DECODE_INPLACE=1` |
| `stack` | `dd1d104148` | the five switches of the table, without `TUNABLEOP_PREFER` |
| `tunable` | `b2d569c402` | `TUNABLEOP_PREFER=rocm7.14-rocblas5.5-hi2-27b-tp2` |
| `final` | `b2d569c402` | the five switches and `TUNABLEOP_PREFER` (this is what the recipe sets, with `VLLM_RDNA_LM_HEAD_GEMV_MAX_M=32`, which the recipe already carried) |
| `final-mtp2` | `b2d569c402` | `final` and `MTP=2` |
| `mtp2-nofix`, `mtp2-lmfix` | `f7edb8b1e7` | `MTP=2` and `VLLM_RDNA_LM_HEAD_GEMV_MAX_M=8` or `32` |
| `conc-on`, `conc-off` | `b2d569c402` | `VLLM_RDNA_AR_HIER=0` (and `TUNABLEOP_PREFER=` for `conc-off`) |
| `gemm` | `b47eb6c384` | `RDNA_HINT_FLEX_PREFILL_GEMM=hip` and `TUNABLEOP_PREFER=rocm7.14-rocblas5.5-hi2-27b-tp2` (HIP 2,3) |
| `final-a2` | `b47eb6c384` | as `final`, on HIP 0,1 |
| `final2-a2` | `b47eb6c384` | as `final-a2` and `RDNA_HINT_FLEX_PREFILL_GEMM=hip` |
| `final2-mtp2-a2` | `b47eb6c384` | as `final2-a2` and `MTP=2` |
| `ship-a2` | `15a4bc6f49` | none: the recipe `27b-hi2-tp2` as it was at that commit (seven switches, no dense GEMM) |
| `ship-mtp2-a2` | `15a4bc6f49` | `RECIPE=27b-hi2-tp2-mtp2`, nothing else |
| `ship-gemm-a2` | `15a4bc6f49` | `RDNA_HINT_FLEX_PREFILL_GEMM=hip` over the recipe `27b-hi2-tp2`: the recipe as flipped. Pending |

The scoreboard tools were synced to the box tree of `base` and `attn` as files (its `tree_sha`
reads `ef0b99ea9b+scoreboard`). The box trees of `gemm`, `final-a2`, `final2-a2` and
`final2-mtp2-a2` are copies of `b47eb6c384` without `.git` (`tree_sha` reads `b47eb6c384` for
`gemm` and `unknown` for the three `-a2` arms); `ship-a2` and `ship-mtp2-a2` ran from a copy of
`15a4bc6f49`. The a2 arms used the `serve_validate.sh` of commit `de9f74c240` (branch
`feat/hi2-c8-opt`, not in this one), which only adds a lock that serializes the measured
windows of the card pairs. The model snapshot is `db2a3aa3cd5efb0f92403cd4864e1f0d99378149`.

## Dense HIP prefill GEMM

`RDNA_HINT_FLEX_PREFILL_GEMM=hip` replaces the dequant plus rocBLAS GEMM of a prefill chunk by
a dense HIP kernel above M = 48, per shape where its dispatch table measured faster and
rocBLAS everywhere else
([`hi2-prefill-gemm-2026-11-15.md`](hi2-prefill-gemm-2026-11-15.md),
[`hi2-prefill-gemm2-2026-11-15.md`](hi2-prefill-gemm2-2026-11-15.md)). The code is off by
default and both recipes turn it on. The kernel and composite microbenchmark gives 1.09 to
1.12x over tuned rocBLAS at M = 2048 and 1.16x over the rocBLAS heuristic (geomean of 45
cells) [measured].

The gate for making it a default was set before the arms ran [inference: set by the author,
not agreed outside the project]: lower TTFT on the 16k and 32k cells, no loss on the 1k cells,
long PPL and prompt KL at `final`'s values (3.6635, 7.7e-6), every validity line PASS.
`final2-a2` against `final-a2` (same card pair) meets it: TTFT 21.81 to 20.45 s at 16k/1k c=1
and 47.43 to 44.77 s at 32k/2k c=1; 1k/512 c=1 40.7 to 40.9 tok/s and c=8 137.7 to 147.7;
long PPL 3.6631 (3.6635) and prompt KL 7.80e-6 (7.73e-6), a rounding-level move; KFD
evictions 0 and every probe PASS. The MTP recipe gains the same way (`final2-mtp2-a2` against
`ship-mtp2-a2`, acceptance PASS), and so does the switch with the tuned rows on the other pair
(`gemm` against `tunable`).

`ship-gemm-a2` is the confirming arm of the flip: the recipe of `15a4bc6f49` with the switch
set from the environment, which is the recipe as it is now. Result: pending.

## Related notes

- [`rdna-hint-flex-weights.md`](rdna-hint-flex-weights.md): format, loader, kernels, compute modes.
- [`hi2-27b-tp2-2026-11-14.md`](hi2-27b-tp2-2026-11-14.md): the first recipe, probes and the decode kernel-time breakdown.
- [`hi2-attn-prefill-2026-11-14.md`](hi2-attn-prefill-2026-11-14.md), [`hi2-residual-direct-2026-11-15.md`](hi2-residual-direct-2026-11-15.md), [`hi2-gdn-glue-2026-11-15.md`](hi2-gdn-glue-2026-11-15.md), [`hi2-27b-tunableop-2026-11-15.md`](hi2-27b-tunableop-2026-11-15.md): one note per switch, with method and unit evidence.
- [`hi2-27b-mtp-2026-11-14.md`](hi2-27b-mtp-2026-11-14.md): the MTP drafter.
- [`hi2-scoreboard-2026-11-14.md`](hi2-scoreboard-2026-11-14.md): the scoreboard driver and report.
- [`docs/rdna/serving.md`](../rdna/serving.md): launcher keys and the recipe list.
