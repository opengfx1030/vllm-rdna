# W4A8 sdot4 test plan

Two parts. **Part A** runs anywhere without a GPU and is what an agent (or a
reviewer on a laptop) runs before asking for GPU time. **Part B** is the run
on the PR with the V620s, gate by gate.

No fabricated timings: every Part B cell stays blank until measured on
gfx1030 (V620). The kernel stays explore-only; nothing here flips a default,
`can_implement`, or a serve script.

## Part A — agent-side (no GPU)

### Setup

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install numpy pytest regex   # reference, ISA and C ABI checks
uv pip install torch                # optional: harness smoke tests
clang -print-targets | grep amdgcn  # any clang with AMDGPU (distro clang-18 works)
```

### A1. Unit tests

```bash
.venv/bin/python -m pytest benchmarks/kernels/w4a8_sdot4_explore -q
```

| Test | Guards against |
| --- | --- |
| `test_shuffle_matches_exllama_source`, `test_w4a16_dequant_reads_shuffled_pack_in_k_order`, `test_shuffle_roundtrip` | Modelling a different W layout than the one `RDNA2W4A16LinearKernel` leaves in memory |
| `test_swar_unpack_yields_a_perm_order` | A byte order not matching the 3-VALU unpack |
| `test_sign_extension_is_wrong_for_zero_point_packs` | Re-introducing the old sign-extension contract |
| `test_sdot4_matches_bruteforce` | Getting `v_dot4_i32_i8` semantics wrong (signed bytes, −128·−128 lanes) |
| `test_act_quant_matches_vllm_convention` | Rounding or zero-row handling drifting from `dynamic_scaled_int8_quant` |
| `test_tile_layout_matches_kernel_indexing` | The tiled A / Σa layout the kernels index, incl. zero padding rows |
| `test_group_partials_are_exact` (uint4, uint4b8 × G 32/64/128 × K_STEP 16/32) | Any error in unpack + permutation + zero fold: integer partials must equal `Σ a·(q − z)` exactly |
| `test_f32_path_within_bound`, `test_f16_split_k_within_bound` | Flush order, split-K, fp16 output exceeding the error bounds G2 uses |
| `test_short_loop_body_is_caught` | ConfigH: advertising K_STEP=32 while the body consumes 16 must fail the oracle |
| `test_worst_case_partials_stay_exact` | i32 wrap or inexact f32 cast at a=−128, q=15, z ∈ {0, 16}, G up to 4096 |
| `test_budget_numbers_quoted_in_design`, `test_split_k_is_group_aligned`, `test_eligibility_rules` | Docs and rules drifting from the model |
| `test_a8_error_grows_with_outliers` | Documents why G1 exists |
| `test_rdna2_w4a16_bakes_a_rounded_bias` | Losing the bit-exact model of the gfx1030 W4A16 dequant that the G2 baseline columns and the G1 reading rely on ([PRIOR-ART](PRIOR-ART-RDNA3.md) §1) |
| `test_per_group_a_scales_stay_exact_and_bounded`, `test_per_group_a_scales_contain_outliers` | `*_ag` math: exact integer partials, one more rounding in the bound; and the reason for the variant |
| `test_isa_matches_budget` | See A2 (skipped without an AMDGPU clang) |
| `test_capi_glue_compiles` | The ctypes glue no longer compiling, or a kernel missing from it |
| `test_harness_cpu.py` (needs torch) | `bench.py` plumbing: buffer sizes (per-token and per-group scales), layouts, bounds, report flow, `--baseline`, `--split-k`, G1 hooks in both scale modes; a kernel with the wrong zero point must fail `check`, one that writes garbage must fail `bench` |

### A2. ISA audit

```bash
.venv/bin/python -m benchmarks.kernels.w4a8_sdot4_explore.isa_check --markdown isa.md
```

Compiles the device header for gfx1030 (plain clang + `w4a8_sdot4_isa_shim.h`)
and the C ABI glue on both passes against a HIP stub. Exits non-zero if any
config/group has: a `v_dot4` count per group other than `2·MT·NPT·G/8`
(ConfigH at the ISA level), scratch, SGPR spills, quarter-rate multiplies in
the loop, or A not coming from LDS (kLds) / scalar loads (kSmem). The loop
block excludes fall-through preheader and exit blocks (`; %bb.N`).

Recorded with Ubuntu clang 18.1.3 (per thread per weight group; the last
numeric column is the W4A16 budget divided by compiled W4A8 VALU):

| config | G | VGPR | SGPR | occ | v_dot4 (want) | other VALU (budget) | v_mov | ds_read | s_load | W4A16/W4A8 VALU | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| a16_lds_k32 | 32 | 160 | 40 | 6 | 512 (512) | 281 (240) | 6 | 40 | 0 | 1.47 (model 1.55) | ok |
| a16_lds_k32 | 64 | 160 | 40 | 6 | 1024 (1024) | 337 (288) | 6 | 72 | 0 | 1.72 (model 1.78) | ok |
| a16_lds_k32 | 128 | 160 | 38 | 6 | 2048 (2048) | 456 (384) | 13 | 136 | 0 | 1.87 (model 1.92) | ok |
| a16_lds_k16 | 32 | 156 | 40 | 6 | 512 (512) | 281 (240) | 6 | 40 | 0 | 1.47 (model 1.55) | ok |
| a16_lds_k16 | 64 | 160 | 40 | 6 | 1024 (1024) | 337 (288) | 6 | 72 | 0 | 1.72 (model 1.78) | ok |
| a16_lds_k16 | 128 | 160 | 40 | 6 | 2048 (2048) | 456 (384) | 13 | 136 | 0 | 1.87 (model 1.92) | ok |
| a16_smem_k16 | 32 | 163 | 92 | 5 | 512 (512) | 279 (240) | 4 | 0 | 9 | 1.48 (model 1.55) | ok |
| a16_smem_k16 | 64 | 163 | 92 | 5 | 1024 (1024) | 335 (288) | 4 | 0 | 17 | 1.72 (model 1.78) | ok |
| a16_smem_k16 | 128 | 166 | 98 | 5 | 2048 (2048) | 447 (384) | 4 | 0 | 33 | 1.87 (model 1.92) | ok |
| a8_smem_k32 | 32 | 104 | 102 | 9 | 256 (256) | 183 (144) | 4 | 0 | 5 | 1.49 (model 1.64) | ok |
| a8_smem_k32 | 64 | 105 | 100 | 9 | 512 (512) | 239 (192) | 4 | 0 | 9 | 1.75 (model 1.86) | ok |
| a8_smem_k32 | 128 | 105 | 100 | 9 | 1024 (1024) | 351 (288) | 4 | 0 | 17 | 1.91 (model 2.00) | ok |
| a8_lds_k32 | 32 | 91 | 40 | 10 | 256 (256) | 185 (144) | 6 | 20 | 0 | 1.49 (model 1.64) | ok |
| a8_lds_k32 | 64 | 91 | 40 | 10 | 512 (512) | 241 (192) | 6 | 36 | 0 | 1.74 (model 1.86) | ok |
| a8_lds_k32 | 128 | 91 | 40 | 10 | 1024 (1024) | 353 (288) | 6 | 68 | 0 | 1.91 (model 2.00) | ok |
| a32n2_lds_k32 | 32 | 163 | 40 | 5 | 512 (512) | 240 (216) | 2 | 80 | 0 | 1.46 (model 1.51) | ok |
| a32n2_lds_k32 | 64 | 172 | 41 | 5 | 1024 (1024) | 288 (240) | 18 | 144 | 0 | 1.67 (model 1.73) | ok |
| a32n2_lds_k32 | 128 | 172 | 41 | 5 | 2048 (2048) | 480 (288) | 146 | 272 | 0 | 1.73 (model 1.88) | ok |
| c16_lds_k32 | 32 | 160 | 40 | 6 | 512 (512) | 281 (240) | 6 | 40 | 0 | 1.47 (model 1.55) | ok |
| c16_lds_k32 | 64 | 160 | 40 | 6 | 1024 (1024) | 337 (288) | 6 | 72 | 0 | 1.72 (model 1.78) | ok |
| c16_lds_k32 | 128 | 160 | 38 | 6 | 2048 (2048) | 456 (384) | 13 | 136 | 0 | 1.87 (model 1.92) | ok |
| a16_lds_k32_ag | 32 | 158 | 38 | 6 | 512 (512) | 346 (304) | 7 | 48 | 0 | 1.36 (model 1.43) | ok |
| a16_lds_k32_ag | 64 | 158 | 38 | 6 | 1024 (1024) | 402 (352) | 7 | 80 | 0 | 1.64 (model 1.70) | ok |
| a16_lds_k32_ag | 128 | 160 | 45 | 6 | 2048 (2048) | 528 (448) | 21 | 144 | 0 | 1.81 (model 1.87) | ok |
| a8_lds_k32_ag | 32 | 91 | 38 | 10 | 256 (256) | 218 (176) | 7 | 24 | 0 | 1.38 (model 1.52) | ok |
| a8_lds_k32_ag | 64 | 91 | 38 | 10 | 512 (512) | 274 (224) | 7 | 40 | 0 | 1.67 (model 1.78) | ok |
| a8_lds_k32_ag | 128 | 91 | 38 | 10 | 1024 (1024) | 386 (320) | 7 | 72 | 0 | 1.86 (model 1.95) | ok |

Act quant (MT 8/16/32): 31 VGPR per token, 28 per group, occupancy 16, no
scratch. Probes: 8 `v_dot4` / `v_dot2` / `v_fma` per loop block as intended.
`a32n2_lds_k32` at G=128 is the known weak spot (146 `v_mov`, 1.73 vs 1.88).
The `*_ag` rows are the per-token rows plus one `v_mul_f32` per output and
group (64 at MT=16, NPT=4) and the scale reads.

### A3. The explore stays unwired

```bash
git grep -n -I -e w4a8_sdot4 -e W4A8_SDOT4 -e explore_w4a8 -- \
    CMakeLists.txt cmake csrc/rocm/torch_bindings.cpp csrc/rocm/ops.h vllm scripts setup.py
git grep -n -I "rocm/explore" -- ':!docs' ':!benchmarks' ':!csrc/rocm/explore'
```

Both must print nothing.

### A4. Lint

```bash
pre-commit run --files docs/explore/w4a8-sdot4/*.md csrc/rocm/explore/* \
    benchmarks/kernels/w4a8_sdot4_explore/*.py
```

ruff, clang-format 21.1.2, markdownlint, typos. The Python lives under
`benchmarks/`, so the mypy hook does not cover it.

### A5. Rules for an agent changing the draft

- Change a layout or the math: update `reference.py` and its tests first,
  then the kernel. Never widen a bound to make a check pass.
- Change the kernel: rerun A2 and paste the table into the PR. New sweep
  configs go in `W4A8_EXPLORE_CONFIGS` (the `.so` and the audit pick them up).
- No torch binding, CMake entry or dispatcher change inside the explore.
- Never write a timing into these docs that was not measured on gfx1030.

## Part B — V620 run (on the PR)

### Prerequisites

- A gfx1030 box with the PR checked out and built (the baseline needs
  `_rocm_C.gptq_gemm_rdna2_prefill`, `_C.gptq_shuffle`, `_C` int8 quant).
- hipcc from the same ROCm major as `torch.version.hip`, so one HIP runtime is
  loaded (`HIPCC=...` or `--hipcc`).
- One GPU per run: `HIP_VISIBLE_DEVICES=0`.
- Record once per run: git SHA, `torch.__version__`, `torch.version.hip`,
  `hipcc --version`, board, power cap (180 W today), clocks from `amd-smi`.
- Results go to `docs/explore/w4a8-sdot4/RESULTS-<date>.md` with the JSON
  files; update the README status line with each gate's verdict.

### B0. Build and audit with the real compiler

```bash
python -c "from benchmarks.kernels.w4a8_sdot4_explore import lib; print(lib.build(save_temps=True))"
python -m benchmarks.kernels.w4a8_sdot4_explore.isa_check \
    --clang "$ROCM_PATH/llvm/bin/clang" --markdown isa-rocm.md
```

Pass: A2 is clean with ROCm's clang too. Note any VGPR/occupancy drift from
the clang-18 table.

### B1. G0 — issue rate

```bash
python -m benchmarks.kernels.w4a8_sdot4_explore.bench peak --json g0.json
```

Eight independent chains per lane, 16 blocks of 256 per multiprocessor.

| Kind | µs | T lane-instr/s | T MAC/s |
| --- | --- | --- | --- |
| `v_fma_f32` | | | |
| `v_dot2_f32_f16` | | | |
| `v_dot4_i32_i8` | | | |

| sdot4/fdot2 MAC ratio | implied clock (GHz) | `amd-smi` sclk under load | verdict (≥ 1.8) |
| --- | --- | --- | --- |
| | | | |

### B2. G0.5 — what bounds W4A16 today (informational)

Profile the production W4A16 op on the large-M cells with the venv-bundled
`rocprofv3` (the only one that works on this box, see
`docs/profiling/2026-09-10-decode-kernel-profile.md` §1). Counter names
differ on gfx10 (L2 is `GL2C_*`), so list them first:

```bash
rocprofv3 -L > counters.txt   # pick names from here
rocprofv3 --pmc SQ_WAVES SQ_INSTS_VALU SQ_INSTS_LDS SQ_WAVE_CYCLES \
    --output-format csv -d prof-g05 -- \
    python -m benchmarks.kernels.w4a8_sdot4_explore.bench bench \
    --configs a16_lds_k32 --iters 3 --warmup 1
```

| Cell (M×N×K) | W4A16 kernel | µs | VALU instr / wave | VALU instr / (wave·cycle) | L2 / IC hit % | Reading |
| --- | --- | --- | --- | --- | --- | --- |
| 624×6144×2560 | | | | | | |
| 2048×6144×2560 | | | | | | |
| 2048×2560×8704 | | | | | | |

If W4A16 is far from VALU-bound, halving its VALU count will not halve its
time: lower the G3 expectation before tuning.

Also confirm that the baseline really issues `v_dot2_f32_f16`: the W4A16
budget in DESIGN §5 assumes it, and on gfx1100 hipcc did not form it from
`__hfma2` code. Disassemble the gfx1030 code object of `_rocm_C` (ROCm's
`roc-obj` tools) and count `v_dot2_f32_f16` against `v_pk_fma_f16` in
`gemm_dynamic_kernel`.

### B3. G1 — accuracy of int8 FFN inputs (no new kernel)

```bash
python -m benchmarks.kernels.w4a8_sdot4_explore.fakequant_eval \
    --model /models/Qwen3.8-27B-AWQ-INT4 --tp 2 --max-model-len 4096 \
    --ppl-file <text> --ppl-ctx 2048 --ppl-windows 16 --gsm8k 500 --json g1.json
# prefill-only (decode rows stay W4A16), and per layer if the combined run fails:
... --min-rows 257
... --layers mlp.gate_up_proj
... --layers mlp.down_proj
# per-(token, group) scales, the *_ag configs (G = the checkpoint's group size):
... --act-group-size 32
```

Use wikitext-2 test text for `<text>` if it is on the box
(`benchmarks/sonnet.txt` otherwise; only the delta matters). Run under the
serving environment; the engine is eager because hooks do not run inside
captured graphs, and the script aborts if no hook fired.

The baseline is the production W4A16 path, which carries the baked fp16
bias ([PRIOR-ART](PRIOR-ART-RDNA3.md) §1). Fake quant adds A8 on top of it,
while the W4A8 kernel's weight term is exact. The delta therefore overstates
the W4A8 cost: a pass is conservative, a marginal fail is not final.

| Model | Layers | min rows | A scales | PPL W4A16 | PPL + int8 FFN | Δ | GSM8K W4A16 | GSM8K + int8 FFN | Δ | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3.8-27B-AWQ-INT4 (G32) | gate_up + down | 0 | token | | | | | | | |
| Qwen3.8-27B-AWQ-INT4 (G32) | gate_up + down | 257 | token | | | | | | | |
| Qwen3.8-27B-AWQ-INT4 (G32) | gate_up + down | 0 | token, G=32 | | | | | | | |
| Qwen3.8-27B-AWQ-INT4 (G32) | gate_up | 0 | token | | | | | | | |
| Qwen3.8-27B-AWQ-INT4 (G32) | down | 0 | token | | | | | | | |

Pass (proposed): PPL ≤ +2 % relative and GSM8K ≥ −1.0 point. If only the
per-group row passes, shape (1d) becomes the candidate.

### B4. G2 — correctness

```bash
python -m benchmarks.kernels.w4a8_sdot4_explore.bench check --json g2.json
python -m benchmarks.kernels.w4a8_sdot4_explore.bench check --quick   # fast re-run
```

Covers edge tails, the prefill and decode cells, both formats, G 32/64/128,
every sweep config (per-group scales for `*_ag`); f32 output (split 1)
against `f32_flush_bound`, fp16 output with the auto split against
`f16_output_bound`. The fp16 case runs twice. `same bits ×2 = no` means the
pk4 CAS epilogue is order-dependent on this board, which graduation must fix
(DESIGN §10). It also checks that `gptq_shuffle` equals the reference
shuffle and that the C and Python split-K rules agree.

| Check | Result |
| --- | --- |
| `gptq_shuffle` == `reference.exllama_shuffle` | |
| split-K C/Python mismatches | |
| act quant bit-exact vs reference (all rows, both scale modes) | |
| act quant == vLLM `scaled_int8_quant` | |
| W4A8 rows passing (of total) | |
| worst f32 err/bound, worst fp16 err/bound | |
| fp16 split-K rows with the same bits twice (of total with split > 1) | |

The baseline table compares the production W4A16 op with exact dequant and
with the bit-exact model of its baked bias. Expected on this random data:
op vs baked emulation near the fp16 output rounding (≲ 1e-3), op vs exact
about 2–3e-2. If op vs emulation is about 1e-2, that op does not dequant as
`qdq_4_rdna2.cuh` does: note which one, since the G1 reading depends on it.

| Cell | fmt | G | W4A16 op | op vs exact | baked emu vs exact | op vs baked emu | W4A8 per token vs exact | W4A8 per group vs exact |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 624×6144×2560 | uint4 | 32 | | | | | | |
| 624×6144×2560 | uint4b8 | 128 | | | | | | |
| 2048×2560×8704 | uint4 | 32 | | | | | | |

Any failure blocks B5. The K_STEP rules are covered by A1
(`test_short_loop_body_is_caught`) and A2 (exact `v_dot4` counts); K % 32 and
group alignment are rejected by the C ABI (`w4a8_error_str`).

### B5. G3 — microbench against the production W4A16 path

```bash
M=benchmarks.kernels.w4a8_sdot4_explore.bench
# 27B AWQ (uint4, G=32); --baseline auto asks the production selector
python -m $M bench --cells prefill --group-size 32 --weight-type uint4 --json g3-g32.json
python -m $M bench --cells prefill --group-size 32 --weight-type uint4 --cold --json g3-g32-cold.json
# GPTQ / AutoRound (uint4b8, G=128): production uses exllama above M=256
python -m $M bench --cells prefill --group-size 128 --weight-type uint4b8 --json g3-g128.json
# no split-K (JartX's rule: the large-M grids already oversubscribe the GPU)
python -m $M bench --cells prefill --group-size 32 --weight-type uint4 --split-k 1 --json g3-g32-split1.json
# kernel-to-kernel against ConfigA, and the informational sets
python -m $M bench --cells prefill --baseline prefill --w4a16-force-config 1 --json g3-configA.json
python -m $M bench --cells k-sweep,decode --group-size 32 --json g3-extra.json
```

Every timing row also carries `rel-L2 vs W4A16`: the W4A8 output left by the
timed calls against the W4A16 op's output. Above 0.1 the row is marked BAD
and the run exits 1. About 3e-2 is expected on random data, mostly the W4A16
bias. On gfx1100 a corrupted engine benchmarked faster, so no speedup is
recorded without this check.

Fill with the best W4A8 config per cell (all configs are in the JSON):

| M | N | K | Note | W4A16 op | W4A16 µs | Best W4A8 config | Split | Act quant µs | W4A8 GEMM µs | × GEMM | × total | × total, split 1 |
| ---: | ---: | ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 32 | 2560 | 8704 | down-proj class | | | | | | | | | |
| 96 | 2560 | 8704 | ConfigA ≥96 band | | | | | | | | | |
| 128 | 6144 | 2560 | microbench M=128 | | | | | | | | | |
| 256 | 6144 | 2560 | M=256 boundary | | | | | | | | | |
| 624 | 1024 | 2560 | small-N | | | | | | | | | |
| 624 | 6144 | 2560 | microbench mid-M | | | | | | | | | |
| 624 | 8704 | 2560 | TP=2 intermediate | | | | | | | | | |
| 624 | 12288 | 2560 | microbench high-N | | | | | | | | | |
| 1856 | 6144 | 2560 | large-M profile band | | | | | | | | | |
| 2048 | 2560 | 8704 | full-chunk down | | | | | | | | | |
| 2048 | 6144 | 2560 | microbench M=2048 | | | | | | | | | |
| 2048 | 8704 | 2560 | full-chunk intermediate | | | | | | | | | |

Pass (proposed): × total ≥ 1.25 on at least 3 of the 4 cells with M ≥ 624
and N ≥ 6144 at G=32 (≥ 1.4 at G=128), hot and cold. The smallest M where
× total > 1 becomes the dispatch threshold. Decode cells are expected to lose
(README shape 3); record them anyway.

### B6. Counters on the winner

Repeat B2 for the best W4A8 config next to W4A16 on the same three cells:
VALU instructions per wave should drop roughly as A2 predicts; if time does
not follow, the L2 / Infinity Cache columns say whether W re-reads bound it
(DESIGN §7), which points at `a32n2_lds_k32` or an M-fastest raster.

| Cell | Kernel | µs | VALU instr / wave | L2 / IC hit % | Achieved W GB/s |
| --- | --- | --- | --- | --- | --- |
| 2048×6144×2560 | W4A16 | | | | |
| 2048×6144×2560 | W4A8 best | | | | |

### B7. G4 — end to end (separate PR, only after G3 passes)

Wire behind `VLLM_RDNA2_W4A8_SDOT4=0` per DESIGN §10, then run the
`docs/rdna2/bench_27b_awq_matrix.md` matrix (1k/512 and 16k/1k, c=1/4/8)
and GSM8K through `tests/evals/gsm8k` with the flag on and off. Pass: prefill
tok/s +10 % at 16k/1k c=1 with G1-level accuracy.

## Graduation checklist (carried over; for the wiring PR)

Graph hygiene, same class as the GDN / EXL3 capture bugs:

| Item | Pass? |
| --- | --- |
| Scratch and outputs allocated with zeros, not `empty`, where atomics accumulate | |
| Out tensors marked mutating (`Tensor!`) in the binding | |
| No `.item()` / D2H under capture | |
| No host `printf` of device scalars on the capture stream | |
| Persistent / immortal buffers for graph-captured launches (`rdna2_graph_keepalive.cuh`) | |
| Default path still does not register the op unless the env flag is set | |

From JartX's gfx1100 work ([PRIOR-ART](PRIOR-ART-RDNA3.md)):

| Item | Pass? |
| --- | --- |
| Split-K epilogue deterministic (f32 partials + fixed-order reduce), or split 1 | |
| M threshold inside the C++ op entry; Python dispatches on static facts only | |
| W4A8 in its own TU; W4A16 kernels' ISA identical before and after | |
| Soak run with an output checker proven on a known-bad sample, in the same engine start as the benchmark | |

Activation path (GEMM A8 only):

| Check | Pass? |
| --- | --- |
| The GEMM consumes int8 A produced by the act-quant op (no fp16→i8 inside the dot loop) | |
| No second act quant for GDN state | |
| No requant of KV / indexer to A8 | |
| No silent requant of leftovers / embed / head / norms | |
| Only eligible dense FFN layers on W4 packs (`reference.eligibility`) | |
