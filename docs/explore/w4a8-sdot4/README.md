# Explore: W4A8 sdot4 — ranked shapes (nibble→i8)

**Status**: Explore-only research. **Not dest.** Not UNC-26. Not a default
serve path. Nothing here is built by CMake, registered as a torch op, or
reachable from `can_implement` / serve scripts.

**Card**: Project 4 — *Explore: W4A8 sdot4 — ranked shapes (nibble→i8)*
(V620 inference board). Dest remains
[UNC-26](https://linear.app/uncoolred/issue/UNC-26/finish-exl3-hip-3inst-fdot2)
(EXL3 HIP 3inst → fdot2).

W4A8 here means **W uint4 × A int8 → `v_dot4_i32_i8`**
(`__builtin_amdgcn_sdot4`, emitted as `v_dot4c_i32_i8`) on **gfx1030**, for
the GEMM activations of dense FFN layers that already sit on W4 packs.

- Draft of what it should be: [DESIGN.md](DESIGN.md)
- Agent-side checks and the V620 run: [TESTPLAN.md](TESTPLAN.md)
- JartX's RDNA3 kernels and what they change here:
  [PRIOR-ART-RDNA3.md](PRIOR-ART-RDNA3.md)
- First V620 pass, review and next round:
  [RESULTS-2026-09-28.md](RESULTS-2026-09-28.md)

| Gate (2026-09-28) | Status |
| --- | --- |
| G0 issue rate | **PASS**: sdot4/fdot2 MAC ratio 2.01 |
| G1 accuracy | dense 1.5B: per token **FAIL** (+10.77 % PPL), per (token, group) **PASS** (+0.72–0.75 %); 27B target not run yet |
| G2 correctness | small cells pass; production cells to re-run with the fixed harness |
| G3 microbench | 1.75–2.9× hot, preliminary; per-group configs still to be timed |

## TL;DR

- **Why**: prefill GEMMs are the compute-bound part of serving on the V620;
  decode is launch-bound. gfx1030 has no matrix cores, and
  `v_dot4_i32_i8` is the only instruction that does more MACs per VALU issue
  (4) than the `v_dot2_f32_f16` the W4A16 kernels use (2).
- **Honest ceiling**: every pack we serve has per-group scales, so each group
  costs a 3-VALU flush per output. The main-loop VALU count drops **≈1.5× at
  group 32** (Qwen3.8-27B-AWQ) and **≈1.9× at group 128** (GPTQ / AutoRound),
  not 2×. The budget model and the clang-18 gfx1030 ISA agree
  ([DESIGN §5](DESIGN.md#5-valu-budget)).
- **Risks**: accuracy of int8 FFN inputs with outliers (G1); at twice the MAC
  rate the ConfigA-class W re-reads may bind before VALU does (G3 counters);
  i32 + f32 accumulators cost registers (160 VGPRs, occupancy 6).
- **Kill fast**: G0 (sdot4 issue rate on the real board) and G1 (fake-quant
  accuracy on the real model) need no vLLM wiring and either can end the
  explore before any tuning.
- **Prior art**: JartX's gfx1100 work found that the exllama fp16 dequant
  bakes a rounded bias into every weight. gfx1030's W4A16 does the same.
  On random data that costs 2.3–3.3 % rel-L2, several times the per-token
  A8 error, and W4A8's integer weight term does not have it. The survey
  also brought deterministic split-K, M dispatch in C++, output checks
  beside every timing, and a dense-dequant path that passed golden tests
  and corrupted production output
  ([PRIOR-ART-RDNA3.md](PRIOR-ART-RDNA3.md)).
- **In this PR**: a corrected contract, a draft kernel that compiles for
  gfx1030 and passes an ISA audit, a variant with per-(token, group)
  activation scales, a NumPy oracle (including a bit-exact model of the
  W4A16 baked dequant), agent-runnable CPU tests, a standalone V620
  harness (G0/G2/G3) and a fake-quant accuracy script (G1).

## Why

| Fact | Source |
| --- | --- |
| The large-M prefill GEMM (`gptq_rdna2_prefill::gemm_dynamic_kernel`) is the top kernel: 3.5 ms/call, 22.9 % of GPU time in the c=1 window; the c=8 prefill window keeps the GPU 98 % busy | `docs/profiling/2026-09-10-decode-kernel-profile.md` |
| Decode is launch/overhead-bound (GPU 60–65 % idle); a faster decode GEMM buys ~10 % wall clock | same |
| W4A16 prefill measured 1,347 µs (M=624) and 3,992 µs (M=2048) at N=6144, K=2560: 14.6 and 16.1 TFLOP/s effective (arithmetic on the measured µs; ConfigV1 era) | `docs/profiling/2026-09-10-awq-vs-gptq-prefill-microbench.md` |
| 27B AWQ is uint4, asymmetric, group 32; Flash-Next experts are symmetric INT4, group 128 | microbench doc §1; `docs/rdna2/V620-FP16-PERFORMANCE.md` |

Per CU and clock RDNA2 issues 64 VALU lanes. `v_dot2_f32_f16` does 2 MACs
per lane, `v_dot4_i32_i8` does 4, if both issue at full rate under the
board's 180 W cap. G0 measures that instead of assuming it.

## Hypotheses and kill criteria

Thresholds are proposals for the owner to confirm before the V620 run.

| # | Hypothesis | Gate | Pass | If it fails |
| --- | --- | --- | --- | --- |
| H0 | `v_dot4_i32_i8` sustains ≥ 1.8× the MAC rate of `v_dot2_f32_f16` on the V620 | G0 peak probe | sdot4/fdot2 MAC ratio ≥ 1.8 | stop: no ISA lever |
| — | W4A16 ConfigA is VALU-bound on large-M cells | G0.5 counters | informational | sets the realistic G3 target |
| H1 | int8 FFN inputs cost little accuracy on the target models (G1 is pessimistic: it adds A8 on top of the baked W4A16 error) | G1 fake-quant | PPL ≤ +2 % rel, GSM8K ≥ −1.0 pt | per-(token, G) scales (`--act-group-size`, shape 1d), then `gate_up_proj` only, else stop (smoothing is another explore) |
| H2 | the draft is correct on the board | G2 check | all cells within bound; act quant bit-exact | fix before timing |
| H3 | W4A8 incl. act quant beats the production W4A16 path on large-M prefill | G3 bench, output-checked | ≥ 1.25× on ≥ 3 of the 4 M ≥ 624 cells at G=32 (≥ 1.4× at G=128) | next shape in rank order, else stop |
| H4 | it shows end to end | G4, separate PR, only after G3 | prefill tok/s +10 % at 16k/1k c=1, G1-level accuracy | stays an explore |

## Contract

| Item | Rule |
| --- | --- |
| ISA | `__builtin_amdgcn_sdot4(a, w, acc, clamp=false)`; gfx1030 emits `v_dot4c_i32_i8`: four signed i8×i8 products into i32. |
| W pack | The buffer `RDNA2W4A16LinearKernel` already holds: GPTQ `[K/8, N]` int32 packed along K, then `gptq_shuffle` (slot s of a dword holds K offset `{0,2,4,6,1,3,5,7}[s]`). Shared with the W4A16 decode path; no second copy. |
| Nibble expansion | **Zero-extend**: `w & 0x0F0F0F0F` and `(w >> 4) & 0x0F0F0F0F`, 3 VALU per dword. Values 0..15 are valid signed i8. **Never sign-extend** a uint4 / uint4b8 pack. |
| Zero point | `z = stored + zero_offset` (0 for AWQ `uint4`, 1 for GPTQv1 `uint4b8`), folded into the accumulator init: `acc = −z·Σa` (`v_mul_i32_i24`). |
| A | Per-token symmetric int8, identical to vLLM `dynamic_scaled_int8_quant` (`scale = absmax/127`, `rint(x·127/absmax)`, saturate); `*_ag` configs use the same rounding with one scale per (token, G), tiled `[⌈M/MT⌉][K/G][MT]`. Stored tile-interleaved `[⌈M/MT⌉][K/8][MT][8]`; bytes of each 8-K chunk in order `{0,4,1,5,2,6,3,7}` to match the unpack; rows ≥ M are zero. |
| Σa | int32 per (row, group), written by the act-quant kernel, tiled `[⌈M/MT⌉][K/G][MT]`. |
| Accumulate | i32 **within a group**; at each group end `cf += float(acc) · s[g]` (`v_cvt_f32_i32` + `v_fmac_f32`). Activation scale once, in the epilogue (`*_ag`: in the flush, one `v_mul_f32` more). The flushed value is ≤ 2048·G in magnitude (exact in f32 for G ≤ 8192); intermediates ≤ 3968·G (no i32 wrap). |
| Output | fp16 `[M, N]`: plain store for split_k = 1, else pk4 CAS atomic add into a zero-filled buffer (as ConfigA; addition order varies from run to run). An f32 output (split 1) exists for testing. |
| Alignment | K % 32 = 0, K % G = 0, G ∈ {32, 64, 128}, N % 8 = 0. K splits cover whole groups. |
| Scope | Dense FFN `gate_up_proj` / `down_proj` on uint4 / uint4b8 packs without act-order, prefill only (M threshold decided by G3). Not attention, embed, head, norms, GDN state, KV, indexer; routed experts later. W4A16 leftovers stay W4A16. |
| Gate | Off. Nothing in `VLLM_ROCM_EXT_SRC`, `torch_bindings.cpp`, `can_implement` or serve scripts; the harness builds a standalone `.so`. If it graduates: env flag `VLLM_RDNA2_W4A8_SDOT4`, default `0`. |

## What changed vs the first scaffold

| First scaffold | Problem | Now |
| --- | --- | --- |
| "ConfigA-class, LDS=0 (no A/W tile in LDS)" | ConfigA is `Config<256, 4, 32, 16, 0>`; the `0` is **LDS_PAD**. ConfigA stages the whole fp16 A split in LDS (`block_a[M_TILE][k_per_split + LDS_PAD]`). | Shape (1) keeps ConfigA's structure with int8 A in LDS; "no LDS" is its own variant (1b), A through the scalar cache. |
| "sign-extend, not uint4−8" | The packs are uint4 with zero points. Sign extension yields `q − 16·[q ≥ 8]`, which equals `q − z` for no z. | Zero-extend, fold `z·Σa`. |
| "K-contiguous nibbles", LSB first | In memory the pack is exllama-shuffled by `gptq_shuffle`. | SWAR unpack of the shuffled dword; the act-quant kernel writes A in the matching byte order. A is permuted, not W, because W is shared with decode. |
| "i32 through K, scales in the epilogue only" | Impossible with per-group scales, and every pack we serve is grouped. | i32 within a group, f32 across groups; only the activation scale is in the epilogue. This flush is what caps G=32 at ≈1.5×. |
| Per-nibble shift/mask unpack | ~20 VALU per dword. | 3 VALU per dword. |
| Non-gfx1030 `sdot4` returned `acc` unchanged | Silently wrong numbers on another device. | Device stubs plus a host arch check that returns an error code. |
| A read per thread from global; no split-K | No reuse of A across the N tile; small grids at small M. | LDS staging, group-aligned split-K, pk4 atomic epilogue. |
| torch launcher in an unwired `.cu` | Could not be built or run standalone. | C ABI `.so` built by hipcc and driven through ctypes. |
| Test tables without format / group columns, no oracle | Could not tell a layout bug from a rounding difference. | NumPy oracle with error bounds; tables carry format and G. |

## Ranked shapes (research order)

Measure in this order. Names refer to `W4A8_EXPLORE_CONFIGS` in
`csrc/rocm/explore/w4a8_sdot4.cuh`; all are instantiated for G = 32, 64, 128.

### (1) ConfigA-class, int8 A in LDS — measure first

`a16_lds_k32` (and `a16_lds_k16`): THREADS=256, NPT=4 (N_TILE=1024),
M_TILE=16, K_STEP ∈ {16, 32}, group-aligned split-K, pk4 atomic epilogue.
ISA per group at G=32: 512 `v_dot4c` plus 281 other VALU (budget 240),
160 VGPRs, occupancy 6.

### (1b) "LDS=0": A through the scalar cache

`a16_smem_k16`, `a8_smem_k32`. Viable only with the tile-interleaved A
layout (one base pointer, immediate offsets) and a scheduling fence between
K steps; without them the ISA audit showed 400–2,800 SGPR spill ops per
group. Measure next to (1): it trades LDS staging and a barrier for SGPR
pressure.

### (1c) Register and traffic variants

`a8_lds_k32` (M_TILE=8: 91 VGPRs, occupancy 10, twice the W re-reads),
`a32n2_lds_k32` (M_TILE=32, NPT=2: half the W re-reads), `c16_lds_k32`
(ConfigC-class, small N).

### (1d) Per-(token, G) activation scales — the main line since G1

G1 on a dense 1.5B failed per token and passed per group
([RESULTS-2026-09-28.md](RESULTS-2026-09-28.md)), so these configs are what
W4A8 would ship; the per-token ones remain as timing references.

`a16_lds_k32_ag`, `a8_lds_k32_ag`, and `a8_smem_k32_ag` (A through scalar
loads, no LDS, so it can run split 1 at any K): one A scale per row and
weight group,
applied in the group flush. An outlier channel then coarsens only its own
group: on the reference's outlier problems the A8 error drops from 5.3–6.0 %
to 1.0 % (G=32) and 1.9 % (G=128). It is llama.cpp's Q8_1 layout, a scale
and a sum per 32 values. Cost: one `v_mul_f32` per output and group, a
model ratio of 1.43 instead of 1.55 at G=32 and 1.87 instead of 1.92 at
G=128 (compiled 1.36 and 1.81). The act-quant kernel needs no block
reduction for it.

### (2) Recipe-9 LDS 64×64×64 i8 — only if (1)/(1c) lose while compute-bound

WG=256, `64×64×64` i8 tile, ~8 KiB LDS. The flush is per output, so a 2D
tile does not move the G=32 VALU ceiling; it only cuts W and A traffic.
Open it only if G3 counters show (1)/(1c) compute-bound and still losing.

### (3) Decode skinny M ≤ 4 — demoted

At M ≤ 4 the W4 bytes dominate and do not change with A precision, decode is
launch-bound, and W4A8 adds an act-quant launch. Record `no upside` unless G0
surprises. If revisited: only where A is already int8, compared against
`q_gemm_rdna2.cu` / `skinny_gemms_int4.cu`.

### (4) Later: unpack W to an i8 workspace, reuse a W8A8 tile

Transient i8 copy of W per GEMM, then a W8A8 INT8 `sdot4` tile. Needs that
kernel first. Not a dest unpack.

Same class as JartX's dense-dequant + rocBLAS path. That was 1.34× at M=2048
and matched golden, then wrote `!!!!` in production within hours and never
reproduced ([PRIOR-ART-RDNA3.md](PRIOR-ART-RDNA3.md) §5). Before this shape
is considered, it needs a soak harness that has first been shown to fail on
a known-bad build.

## Gates

G0 peak probe → G0.5 baseline counters → G1 fake-quant accuracy →
G2 correctness → G3 microbench → G4 (wiring and end to end, separate PR).
Commands, pass rules and result tables: [TESTPLAN.md](TESTPLAN.md).

## Leave list

Do not land any of these as "the W4A8 path":

- **`sdot8` / `v_dot8_i32_i4`**: W4A4.
- **`sudot4`**: not on gfx1030 (dot8-insts, gfx11).
- **FP8 bit patterns through `sdot4`**: it is integer only.
- **Sign-extended uint4 / uint4b8 nibbles**: wrong numbers.
- **A W4A8-only copy of the weights**: W stays shared with W4A16 decode.
- **`v_mul_lo_u32` or f16 promotion inside the K loop**: quarter rate, or
  precision loss.
- **Unpack W → fp16 then `fdot2` and call it `sdot4`**: that is W4A16 with
  extra copies.
- **Permanent TP ≤ 2 gate**.
- **New checkpoint formats** (QServe-style progressive group quant that keeps
  i32 across groups, SmoothQuant, rotations): a separate explore; this one
  runs on the packs we already serve.

## Dependency

The card orders this explore after a W8A8 INT8 `sdot4` kernel exists (this
tree has W8A8-**FP8** via `fdot2`, not INT8). That still holds for shape (4).
G0 and G1 are shared with the W8A8-INT8 work (same instruction, same
activation quant), so running them first de-risks both.
**UNC-26 EXL3 stays first on dest.** This directory does not bump dest, does
not change EXL3/FA, and does not enable a default kernel.

## Files

| Path | What |
| --- | --- |
| `docs/explore/w4a8-sdot4/DESIGN.md` | Draft of the kernel: math, layouts, budget, registers, traffic, graduation |
| `docs/explore/w4a8-sdot4/TESTPLAN.md` | Agent-side checks (no GPU) and the V620 run, with result tables |
| `docs/explore/w4a8-sdot4/PRIOR-ART-RDNA3.md` | JartX's gfx1100 kernels: inventory, findings, what they change here |
| `docs/explore/w4a8-sdot4/RESULTS-2026-09-28.md` | First V620 pass: gate results, harness review, next-round proposals |
| `csrc/rocm/explore/w4a8_sdot4.cuh` | Device draft: GEMM (10 configs × 3 group sizes), act quant (per token or per group), G0 probes |
| `csrc/rocm/explore/w4a8_sdot4_capi.cu` | C ABI for the harness (hipcc, not CMake) |
| `csrc/rocm/explore/w4a8_sdot4_isa_shim.h` | Lets plain clang compile the device code for the ISA audit |
| `benchmarks/kernels/w4a8_sdot4_explore/reference.py` | NumPy oracle, layout model, budget and split-K rules |
| `benchmarks/kernels/w4a8_sdot4_explore/isa_check.py` | gfx1030 ISA audit and C ABI compile check (clang only) |
| `benchmarks/kernels/w4a8_sdot4_explore/bench.py`, `lib.py` | V620 harness: G0, G2, G3 |
| `benchmarks/kernels/w4a8_sdot4_explore/fakequant_eval.py` | G1 |
| `benchmarks/kernels/w4a8_sdot4_explore/test_*.py` | Agent-runnable tests |
