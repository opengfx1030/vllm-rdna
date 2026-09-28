# W4A8 sdot4 — design draft

**Status**: draft of what the kernel should be, backed by code that compiles
for gfx1030 (`csrc/rocm/explore/w4a8_sdot4.cuh`) and a NumPy oracle
(`benchmarks/kernels/w4a8_sdot4_explore/reference.py`). Not run on a GPU
yet: every performance statement below is either a model or a count of
compiled instructions, never a timing. See [README.md](README.md) for scope
and gates, [TESTPLAN.md](TESTPLAN.md) for how each claim is checked, and
[PRIOR-ART-RDNA3.md](PRIOR-ART-RDNA3.md) for what JartX's gfx1100 kernels
changed here.

## 1. Data flow

```text
x fp16 [M, K]
  │  w4a8_act_quant_kernel<256, MT, PER_GROUP>   one block per M tile
  ▼
a       int8  [T][K/8][MT][8]   T = ceil(M / MT); bytes in A_PERM order; rows >= M zero
a_scale f32   [M]               absmax / 127 (PER_GROUP, *_ag configs: [T][K/G][MT])
asum    int32 [T][K/G][MT]      per-group sums of a
  │
  │  W, unchanged and shared with the W4A16 kernels:
  │    qweight [K/8, N] (gptq_shuffle'd), qzeros [K/G, N/8], scales fp16 [K/G, N]
  ▼
w4a8_gemm_kernel<Cfg>            grid (ceil(N / N_TILE), T, split_k)
  ▼
out fp16 [M, N]                  zero-filled + pk4 CAS atomics when split_k > 1
```

Decode keeps the W4A16 kernels on the same weight buffer; W4A8 only takes
prefill GEMMs above an M threshold (G3 decides it).

## 2. Math

```text
out[m, n] = s_a[m] · Σ_g s[g, n] · ( Σ_{k∈g} a[m, k] · q[k, n]  −  z[g, n] · Σa[m, g] )
```

`*_ag` configs move the activation scale inside the sum, `s_a[m, g]`: same
integers, one more f32 multiply per group.

- `q ∈ [0, 15]` is the stored nibble, zero-extended; `z = stored + zero_offset
  ∈ [0, 16]` (GPTQv1 can reach 16). Sign-extending `q` computes
  `q − 16·[q ≥ 8]`, which is `q − z` for no single `z`; it is only right for
  `z = 8` packs and only after flipping each nibble's top bit.
- The zero term is folded into the accumulator init, `acc = −z · Σa`, with
  `v_mul_i32_i24` (`z ≤ 16`, `|Σa| ≤ 128·G < 2²³`).
- Exactness: the flushed value is `|Σ a(q − z)| ≤ 2048·G` (exact in f32 for
  G ≤ 8192); intermediates stay within `3968·G` (no i32 wrap for any K). The
  only roundings are one f32 FMA per group and the output cast
  (`reference.f32_flush_bound`, `f16_output_bound`).
- Compared with W4A16, which rounds every dequantized weight to fp16, W4A8
  keeps the weight term exact and quantizes the activation instead (step
  `absmax / 127` per token). Accuracy is decided by A (G1), not by the GEMM.
- The gfx1030 W4A16 dequant is also biased. It stores `s·(−1024 − z)` as
  fp16, and that rounding repeats for every weight of a (group, column).
  On random data it costs 2.3–3.3 % rel-L2 against exact dequant (§9,
  [PRIOR-ART-RDNA3.md](PRIOR-ART-RDNA3.md) §1). The baseline W4A8 is
  measured against is not exact.

## 3. Layouts

### 3.1 The existing W pack

`RDNA2W4A16LinearKernel.process_weights_after_loading` packs GPTQ-style
along K and runs `gptq_shuffle`, so each dword holds 8 K values of one column
in exllama order:

| Nibble slot (bits) | 0 (0–3) | 1 (4–7) | 2 (8–11) | 3 (12–15) | 4 (16–19) | 5 (20–23) | 6 (24–27) | 7 (28–31) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| K offset | 0 | 2 | 4 | 6 | 1 | 3 | 5 | 7 |

The fp16 dequant in `qdq_4_rdna2.cuh` reads `(q0,q1) … (q6,q7)` from this
order; `test_w4a16_dequant_reads_shuffled_pack_in_k_order` pins it.

### 3.2 SWAR unpack and the A byte order

```text
lo = w & 0x0F0F0F0F          bytes = q at K offsets (0, 4, 1, 5)
hi = (w >> 4) & 0x0F0F0F0F   bytes = q at K offsets (2, 6, 3, 7)
acc = sdot4(a_lo, lo, acc); acc = sdot4(a_hi, hi, acc)
```

So every 8-byte A chunk is stored as `a[k0 + {0,4,1,5,2,6,3,7}]`. Three VALU
per W dword, shared by the whole M tile, then two `sdot4` per row.

A is permuted rather than W because W is shared with the W4A16 decode path
(a repacked copy would double weight memory), while A is produced by our own
quant kernel on every call, so the permutation costs nothing.

### 3.3 Tile-interleaved A

`[T][K/8][MT][8]`: for each 8-K chunk, the MT rows of a tile are contiguous.

- kLds: staging is a straight 16-byte copy (coalesced reads, conflict-free
  LDS writes); in the loop every A read is base + immediate offset and pairs
  of rows fuse into `ds_read2_b64` (32 instructions per group instead of 64
  `ds_read_b64`).
- kSmem: one scalar base pointer per K step and `s_load_dwordx8/16` with
  immediate offsets. With row-major A, 16 row pointers (32 SGPRs) plus 64
  SGPRs of data spilled.
- Rows past M are zero, so the loop never clamps a row index.

Cost: the act-quant kernel is templated on MT (8, 16, 32). Fusing act quant
into a per-token producer (RMSNorm) later would mean strided writes.

### 3.4 Σa

`[T][K/G][MT]` int32, read once per (row, group) at group start: 16 values
per group at MT=16, which become 8 `ds_read2_b32`.

## 4. Kernel anatomy

```text
grid = (ceil(N / N_TILE), ceil(M / MT), split_k); block = THREADS
k_per_split = K / split_k        split_k divides K/G and is <= 16

kLds: copy this split's A (MT·k_per_split bytes) and Σa (MT·G_split ints) to LDS; barrier
if n >= N: return
cf[MT][NPT] = 0
(nz, s) = params(first group)                        zeros (negated, int16) and f32 scales
for each group gi of the split:
    acc[m][c] = nz[c] · Σa[gi][m]                    v_mul_i32_i24: zero fold as init
    (nz', s') = params(next group, clamped)          prefetch for the next init
    for st in 0 .. G/K_STEP (unrolled; fence between steps for kSmem / NPT=2):
        w[j][c] = W[k/8 + j][n .. n+NPT)  j < K_STEP/8    global_load_dwordx4 per j
        for j: lo, hi = unpack(w[j][c])
               for m: a8 = A[chunk j][row m]                ds_read2_b64 or s_load
                      acc[m][c] = sdot4(a8.x, lo[c], acc[m][c])
                      acc[m][c] = sdot4(a8.y, hi[c], acc[m][c])
    cf[m][c] = fma(float(acc[m][c]), s[c], cf[m][c])  flush (*_ag: float(acc) · s_a[m, gi] first)
    (nz, s) = (nz', s')
epilogue: out[row][n..] = cf · s_a[row]  (*_ag: cf; store if split_k == 1, else pk4 CAS add)
```

The ConfigH lesson is structural: `DW_PER_STEP = K_STEP / 8` is exactly what
the unrolled body consumes, `k` advances by K_STEP, K_STEP ∈ {16, 32} and
`G % K_STEP == 0` are static asserts, and the ISA audit fails unless each
group holds exactly `2·MT·NPT·G/8` `v_dot4`.

Split-K follows ConfigA's `compute_split_k` (LDS budget 16/64/32 KiB by grid
size, then grow while the grid is small or the K range is long), restricted
to divisors of K/G so splits never cut a group. It is written twice
(`w4a8_pick_split_k` in C, `reference.pick_split_k` in Python) and G2 checks
they agree. For example M=624, N=6144, K=2560 picks 10; M=2048, N=6144,
K=2560 picks 4; M=2048, N=2560, K=8704 picks 8 at G=32 and 4 at G=128.

JartX's gfx1100 rule is the opposite: split only while the grid is under
about 2× the resident waves. The V620 holds 864 waves at occupancy 6, and
every M ≥ 624 cell with N ≥ 6144 already launches 2.2–10.7× that. Split-K
costs a zero fill, CAS retries and order-dependent fp16 sums. G3 therefore
also runs `--split-k 1`.

## 5. VALU budget

Per thread and weight group, with M tile MT, NPT columns per thread and
`D = G/8` W dwords per column:

| Work | W4A8 draft | W4A16 ConfigA-class |
| --- | --- | --- |
| Dot products | `2·MT·NPT·D` × `v_dot4` | `4·MT·NPT·D` × `v_dot2` |
| W expansion (shared by the M tile) | `3·NPT·D` (and, lshr, and) | `9·NPT·D` (exllama bit trick, 4 `pk_fma`) |
| Once per group per output | `3·MT·NPT` (i24 mul, cvt, fmac) | none (scale folded into the dequant) |

Per MAC that is `1/4 + 3/G + 3/(8·MT)` VALU for W4A8 against
`1/2 + 9/(8·MT)` for W4A16. The `3/G` term does not depend on the tile: at
G=32 the flush adds 37.5 % to the dot count, which is why group 32 tops out
near 1.5× on existing packs whatever the tile shape.

Model (`reference.loop_budget`):

| MT × NPT | G | W4A8 VALU | W4A16 VALU | Ratio |
| --- | ---: | ---: | ---: | ---: |
| 16 × 4 | 32 | 752 | 1168 | 1.55 |
| 16 × 4 | 64 | 1312 | 2336 | 1.78 |
| 16 × 4 | 128 | 2432 | 4672 | 1.92 |
| 8 × 4 | 32 | 400 | 656 | 1.64 |
| 8 × 4 | 128 | 1312 | 2624 | 2.00 |
| 32 × 2 | 32 | 728 | 1096 | 1.51 |
| 32 × 2 | 128 | 2336 | 4384 | 1.88 |

Compiled (clang 18.1.3, gfx1030, `isa_check.py`), `a16_lds_k32` at G=32, per
thread per group:

| Instruction | Count | Role |
| --- | ---: | --- |
| `v_dot4c_i32_i8` | 512 | dots, exactly `2·16·4·4` |
| `v_mul_i32_i24` | 64 | zero fold as accumulator init |
| `v_cvt_f32_i32`, `v_fmac_f32` | 64 + 64 | group flush |
| `v_and_b32`, `v_lshrrev_b32` | 32 + 17 | unpack (budget 48) |
| other VALU | ~40 | 64-bit W pointer bumps, zero-nibble unpack, scale cvt, 6 `v_mov` |
| `ds_read2_b64`, `ds_read2_b32` | 32 + 8 | A (row pairs), Σa |
| `global_load_dwordx4` | 4 | W: 4 dwords × 4 columns |
| `global_load_dword`, `global_load_dwordx2` | 1 + 1 | next group's zeros, scales |

That is 1.47× fewer VALU than the W4A16 budget (model 1.55×); 1.87× at
G=128 (model 1.92×). The full table for every config and group is in
[TESTPLAN A2](TESTPLAN.md#a2-isa-audit).

Per-(token, G) activation scales add one `v_mul_f32` per output and group,
`1/4 + 4/G + 3/(8·MT)` per MAC:

| Config | G=32 | G=64 | G=128 |
| --- | --- | --- | --- |
| `a16_lds_k32_ag` (16 × 4) | 1.36 (model 1.43) | 1.64 (1.70) | 1.81 (1.87) |
| `a8_lds_k32_ag` (8 × 4) | 1.38 (1.52) | 1.67 (1.78) | 1.86 (1.95) |

Codegen problems the audit caught while drafting, now rules in the kernel:

1. Zeroing the accumulators per group produced 64 `v_mov` per group, because
   the tied `v_dot4c` needs its accumulator in a register. The init is now
   the zero-fold product.
2. Loop-carried int32 zeros lost their 24-bit range and turned the fold into
   64 quarter-rate `v_mul_lo_u32`. Zeros are carried as int16.
3. Row-major LDS A with a runtime stride cost ~77 `v_mov` for addresses.
   Chunk-major A gives immediate offsets and `ds_read2_b64`.
4. SMEM A through row pointers spilled 400–2,800 SGPR lane ops per group.
   Tile-interleaved A plus a scheduling fence between K steps fixed it.

## 6. Registers and occupancy

| Config | G | VGPR | SGPR | Waves/SIMD |
| --- | ---: | ---: | ---: | ---: |
| `a16_lds_k32` | 32–128 | 160 | 38–40 | 6 |
| `a16_smem_k16` | 32–128 | 163–166 | 92–98 | 5 |
| `a8_lds_k32` | 32–128 | 91 | 40 | 10 |
| `a8_smem_k32` | 32–128 | 104–105 | 100–102 | 9 |
| `a32n2_lds_k32` | 32–128 | 163–172 | 40–41 | 5 |
| `c16_lds_k32` | 32–128 | 160 | 38–40 | 6 |
| `a16_lds_k32_ag` | 32–128 | 158–160 | 38–45 | 6 |
| `a8_lds_k32_ag` | 32–128 | 91 | 38 | 10 |
| `a8_smem_k32_ag` | 32–128 | 104–105 | 100–102 | 9 |

The W4A8 loop keeps two accumulator sets, i32 within the group and f32
across groups: `2·MT·NPT` = 128 VGPRs at MT=16, NPT=4, where W4A16 needs 64.
Whether the MT=8 variants' occupancy beats MT=16's W reuse is a G3 question.

## 7. Memory side

- **W re-reads**: every M tile streams the whole pack, `1/(2·MT)` bytes per
  MAC (0.031 at MT=16). At M=2048, N=6144, K=2560 that is 7.9 MB of W and
  1.0 GB streamed at MT=16, 0.5 GB at MT=32
  (`reference.weight_reread_bytes`).
- At MAC rate R the W stream needs `R / (2·MT)` bytes/s; W4A8 doubles R for
  the same bytes, halving the headroom W4A16 had. One FFN weight per rank
  (7.9–11.1 MB) fits the 128 MB Infinity Cache but not the 4 MB L2, so G3
  must record where re-reads are served from. If that is the limit, try
  `a32n2_lds_k32` or an M-fastest raster (§11).
- **A**: staged once per (split, tile); per GEMM `N/N_TILE × M·K` bytes
  (6 × 5.2 MB at M=2048, K=2560, N=6144).
- **Act quant**: reads x twice (the second pass mostly from L2) and writes
  `M·K` bytes plus sums; at M=2048, K=2560 that is 2 × 10.5 MB read and
  5.2 MB written. G3 reports it separately and in the total.

## 8. Act-quant kernel

- One 256-thread block per M tile. Thread `t` owns row `t % MT` and walks
  that row's groups, so group sums need no atomics and neighbouring threads
  write neighbouring 8-byte slots.
- Pass 1: per-row absmax (tree reduction across the threads of a row).
  Pass 2: quantize, reorder to A_PERM, write, sum per group.
- Rounding is `dynamic_scaled_int8_quant`'s: `inv = 127 / absmax` in f32,
  `rint(x · inv)`, saturate. G2 checks it bit for bit against the reference
  and against vLLM's `scaled_int8_quant` op.
- `PER_GROUP` (the `*_ag` configs): a group's absmax only involves the
  thread that quantizes it, so there is no block reduction. It writes one
  scale per (row, group) next to Σa, and the kLds GEMM stages those scales
  into LDS with Σa (8 bytes per row and group instead of 4).
- At graduation: fuse into the producer (RMSNorm before `gate_up_proj`,
  SiLU·mul before `down_proj`) so x is read once.

## 9. Numerics

- Per-token int8 loses little on well-behaved rows but outlier channels
  inflate absmax and crush the rest of the row. On random data (M=32, N=64,
  K=1024, G=32, five seeds) the output rel-L2 against fp16 activations is
  0.76–0.82 % for Gaussian rows and 4.0–4.3 % with 8 of 1024 channels scaled
  20× (`test_a8_error_grows_with_outliers`). Real activations decide; that is
  G1.
- Per-(token, G) scales confine an outlier to its group: 5.3–6.0 % becomes
  1.0 % at G=32 and 1.9 % at G=128 on the prior-art problems
  (`test_per_group_a_scales_contain_outliers`).
- The W4A16 reference point is not exact. Its baked fp16 bias costs
  2.3–3.3 % rel-L2 on the same random data, more than per-token A8 on
  Gaussian rows. G1 adds A8 on top of that bias, so it overstates the W4A8
  cost. G2 compares the W4A16 op with a bit-exact model of the bias
  (`reference.w4a16_rdna2_weights`).
- `down_proj` inputs (after SiLU·mul) are the usual outlier carriers. If G1
  fails with both layers, run it per layer before deciding.
- TP: `down_proj` is row-parallel, so each rank quantizes its own K shard with
  its own per-token scale. G1's hooks do exactly that.

## 10. Graduation (only after G3 passes; separate PR)

1. Torch ops in `csrc/rocm/torch_bindings.cpp` / `ops.h`, sources in the
   gfx1030 block of `VLLM_ROCM_EXT_SRC`; outputs marked `Tensor!`; persistent
   outputs through `rdna2_persist_zeros` / `rdna2_graph_keepalive.cuh`; no
   `.item()` or D2H under capture.
2. Act quant as its own op first; producer fusion afterwards.
3. Dispatch: `rdna2_w4a16.py` decides only from static facts, namely
   `VLLM_RDNA2_W4A8_SDOT4` (default 0) and an eligible layer
   (`reference.eligibility`). The M threshold from G3 goes inside the C++ op
   entry. On gfx1100, a Python branch on `x.size(0)` made Dynamo guard on
   every layer and decode ran 7× slower. W stays the W4A16 buffer; decode is
   unchanged.
4. Deterministic split-K: f32 partials plus a fixed-order reduce (JartX,
   vllm-project/vllm#54706), or split 1 wherever G3 shows it is as fast.
   The output is zero-filled only when split > 1.
5. Own translation unit. Do not touch `qdq_4_rdna2.cuh` or
   `q_gemm_rdna2*.cu`, and diff the W4A16 kernels' ISA before and after.
   On gfx1100, sharing a TU miscompiled another kernel, and growing a shared
   header slowed one.
6. Move the correctness checks to `tests/kernels/quantization/test_rdna2_w4a8.py`
   against the torch op.
7. G4: the 27B AWQ matrix method (`docs/rdna2/bench_27b_awq_matrix.md`) and
   GSM8K through `tests/evals/gsm8k`, with an output check in the same
   engine start as every timing.

## 11. Knobs not in the draft yet

- M-fastest raster, so concurrently resident blocks share W columns in L2.
- Explicit W double-buffering across K steps (registers vs latency).
- An M/N-aware split rule in between ConfigA's and split 1, if G3 shows
  neither end wins everywhere. For W4A16 the AWQ microbench found the
  f32-partials split-K slower than atomics; for W4A8 G3 decides.
- A per-channel (G = K) pack would allow i32 through the whole K and drop the
  flush, but needs requantized checkpoints: out of scope here.
