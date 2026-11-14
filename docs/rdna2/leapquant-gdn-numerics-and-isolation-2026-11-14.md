# LeapQuant GDN: numerics, determinism and state isolation (2026-11-14)

This note covers five safety checks for the LeapQuant GDN state. The upstream
LeapQuant RFC ([vllm#64725](https://github.com/vllm-project/vllm/issues/64725))
lists them as work that a vLLM implementation must do. The note belongs to the
second LeapQuant PR. The first PR is #168 of this fork. Its doc is
[leapquant-gdn-2026-11-13.md](leapquant-gdn-2026-11-13.md).

**Status.** The author had no GPU for this work. The CPU results come from the
FP32 oracle and from host-only checks on a CPU box. Nothing in this note was
run on a GPU. Each GPU claim is *pending GPU validation*. The section "Pending
GPU checklist" lists the open steps. `[INFERENCE]` marks a claim that nobody
measured.

## Credits

- **Item list:** the RFC above, by lihaoyang-amd: sections 2.2 (issues found in
  the prototype), 4 (items 11 to 13, tests and benchmark) and 5 (safety). The
  RFC has no code. No code or text was copied from it or from any other
  project.
- **Re-orthogonalization:** two textbook results. Kahan's "twice is enough"
  rule (B. N. Parlett, *The Symmetric Eigenvalue Problem*, 1980) and the
  dependence test of J. W. Daniel, W. B. Gragg, L. Kaufman and G. W. Stewart
  (*Math. Comp.* 30, 1976). The code is original work for this fork.
- **Benchmark:** `benchmarks/replayssm/e2e_decode_speedup.py` comes from
  [vllm#48018](https://github.com/vllm-project/vllm/pull/48018) by
  Johnny-Liou. This work adds an arm and flags. The file header keeps the
  credit.

## Summary

| # | Item | RFC | CPU-verified | Pending GPU |
|---|---|---|---|---|
| 1 | NaN guard in the fold | 2.2, 4 (11), 5 | 92 oracle tests, old-vs-new probes | kernel build, 3 tests |
| 2 | Determinism under load | 2.2, 4 (11), 5 | nothing (needs a GPU) | 2 tests, 200 repeats each |
| 3 | Stale state of recycled blocks | 5 | 30 oracle tests | 31 tests |
| 4 | Flush race | 2.2, 5 | code review only | 10 tests |
| 5 | End-to-end decode benchmark | 4 (13) | 17 tests, `--help`, `--dry-run` | the benchmark run |

Files:

- `vllm/model_executor/layers/mamba/ops/leapquant_gdn.py`: oracle fix.
- `csrc/rocm/gdn_leapquant_rdna2.cu`: the same fix in the HIP fold. Not built.
- `tests/kernels/mamba/test_gdn_leapquant_oracle.py`: 122 CPU tests.
- `tests/kernels/mamba/test_gdn_leapquant_safety_rdna2.py`: 46 GPU tests. They
  skip without a gfx103x GPU.
- `benchmarks/replayssm/e2e_decode_speedup.py` and
  `tests/benchmarks/test_replayssm_e2e_decode_speedup.py`: the new arm and its
  17 CPU tests.

The tests compare only the header words `[cursor, pending]` (`header[..., :2]`).
They do not read or write header words 2 and 3.

## 1. NaN guard for exactly low-rank states

### The RFC claim

The RFC says that heads whose state is exactly low rank "underflow during the
fold and produce NaN" in the prototype, and that the output became a string of
"!!!!". It asks for two things (section 5). Scale every normalization in the
fold by the largest absolute value and zero out negligible components. Keep the
regression tests. The RFC does not say which normalization failed.
[INFERENCE] It is probably the Gram-Schmidt step of the power iteration, where
a column of a low-rank state cancels to zero.

### Does the old oracle give NaN?

Not for low-rank states. It gives NaN only for a state of about 1e37 or more,
of any rank. Evidence from the CPU runs, with the oracle of PR #168 (commit
`7b4aff7fe3`):

- The new CPU test file (122 tests) ran against that old oracle: 50 tests
  failed and 72 passed. Four failures are NaN checks (`isfinite`): a state of
  1e37 and a state at the float limit, each dense and rank 1. Thirty-five
  check accuracy and orthogonality. Eleven use a header outside the range and
  raise `IndexError` or `AssertionError` (section 3).
- A column of the compensator basis that is rounding noise got a norm above the
  absolute limit `1e-12` and was normalized. The old code did not zero it.
  This happens for rank 1 to 3 at a scale of 1e-3 or more (Table 1).
  The noise column is then not orthogonal to the other columns. The state is
  counted twice in the compensators, and the INT8 residual carries the error.
  This is a loss of accuracy, not NaN.
- The outcome depended on the scale of the state. At scale 1e-6 the noise was
  below `1e-12` and the old code zeroed it. At 1e-3 and above it did not.
- NaN comes from a different place. The smoothing factor is `sqrt(mean |R|)`
  over the `V` rows. The sum overflows to inf when `|R|` is about 1e37 or more.
  Then `z * bscale * csmooth` is `0 * inf = NaN` in every later read. The same
  sum is in the HIP fold.

Table 1 shows `S = a b^T * scale` with random `a`, `b` of rank `r`. "Error" is
`||materialize(store(S)) - S|| / ||S||` in FP64. "Off-diagonal" is the largest
off-diagonal entry of `U~ U~^T` for the stored FP16 compensators. It is 0 for
orthogonal rows.

Table 1: exactly low-rank states, CPU oracle.

| rank | scale | old error | old off-diagonal | new error | new off-diagonal |
|---:|---:|---:|---:|---:|---:|
| 1 | 1e-6 | 5.06e-06 | 0 | 5.06e-06 | 1.4e-04 |
| 1 | 1e-3 | 1.62e-03 | 0.40 | 1.27e-06 | 7.2e-05 |
| 1 | 1 | 1.80e-03 | 0.42 | 1.18e-06 | 4.0e-05 |
| 1 | 1e3 | 2.39e-03 | 0.49 | 1.24e-06 | 4.1e-05 |
| 2 | 1e-3 | 5.04e-04 | 0.15 | 1.54e-06 | 7.9e-05 |
| 2 | 1 | 8.97e-04 | 0.27 | 1.58e-06 | 7.3e-05 |
| 2 | 1e3 | 7.63e-04 | 0.20 | 1.55e-06 | 1.0e-04 |
| 3 | 1e-3 | 1.59e-04 | 0.07 | 1.62e-06 | 1.8e-05 |
| 3 | 1 | 9.44e-04 | 0.42 | 1.69e-06 | 1.8e-05 |
| 3 | 1e3 | 7.99e-04 | 0.35 | 1.66e-06 | 3.8e-05 |

For rank 4 and for dense states the old and new results are the same
(rank 4: 1.9e-06; dense: 6.08e-03). In a rank-1 trajectory (window 16, 40
steps, one fixed key) the decode output has a largest relative error against
FP32 of 1.58e-03 (old) and 5.37e-04 (new).

Table 2 shows the scale sweep. Rank 1 uses `a b^T * scale`. Dense uses
`randn * scale`. All entries of the new column are finite.

Table 2: state scale, CPU oracle, relative error.

| state | old | new |
|---|---|---|
| rank 1, 1e-14 | 5.69e-03 | 5.69e-03 |
| rank 1, 1e-9 | 3.32e-03 | 3.32e-03 |
| rank 1, 1e12 | 3.32e-03 | 3.32e-03 |
| rank 1, 1e36 | 3.32e-03 | 3.32e-03 |
| rank 1, 1e37 | NaN | 3.32e-03 |
| rank 1, max entry 3e38 | NaN | 3.32e-03 |
| dense, 1e-14 | 6.59e-03 | 6.59e-03 |
| dense, 1e12 | 6.54e-03 | 6.54e-03 |
| dense, 1e36 | 6.54e-03 | 6.54e-03 |
| dense, 1e37 | NaN | 6.54e-03 |
| dense, max entry 3e38 | NaN | 6.54e-03 |

The values 3.3e-03 and 6.5e-03 are the INT8 floor. At 1e-9 and below, and at
1e12 and above, the FP16 compensators carry nothing (the FP16 range is 6e-8 to
65504), so the INT8 residual carries the state alone. This limit is part of
the layout and did not change.

### What changed in the oracle

`_orthonormalize` in `leapquant_gdn.py`:

1. Scale each column by its largest absolute value. A column that is zero,
   non-finite or above the float limit is dead (all zero).
2. Sweep twice against the earlier columns (Kahan, "twice is enough").
3. Mark the column dead when the second sweep removed more than half of what
   the first sweep left. Then the rest is rounding noise along the earlier
   columns (dependence test of Daniel, Gragg, Kaufman and Stewart). This is the
   "zero out negligible components" rule of the RFC.
4. Scale again and normalize. Every column of the result is zero or has unit
   norm.

`leapquant_fold_parts`:

- Before, the fixed start block replaced the warm start only when all four
  columns of the previous `comp_k` were zero, and a single dead column stayed
  dead. Now each dead column restarts from its fixed integer column.
- The smoothing factor divides `|R|` by `V` before the sum. Then it cannot
  overflow. For `V` a power of two the result is the same as `mean()` bit for
  bit. A CPU check with `V` = 32, 64, 128 and 256 over 88 state tensors found
  no difference.

Two earlier designs failed on the CPU and were dropped:

- Two sweeps and no dependence test failed for columns that are exact multiples
  of one vector. After two sweeps the residual is again the rounding error along
  the first column.
- A fixed relative threshold on the column norm dropped valid small columns.

`_sane_header` (new, item 3) reads a header that fails `lq_header_ok` as empty,
like the kernels.

### The HIP mirror (not built)

`csrc/rocm/gdn_leapquant_rdna2.cu` has the same logic. **No HIP compiler was
used.** The file header says "NOT BUILT, NOT RUN".

- `lq_orthonormalize`: wave 0 only. It keeps the column in registers, uses the
  new `lq_wmax` and the same two sweeps and test. A NaN entry fails
  `n >= 1.0f`, so the column becomes zero.
- `lq_fold`: a shared flag per column (`alive`, set with `atomicOr` of the
  constant 1) replaces the old all-or-nothing warm start. The result does not
  depend on the order of the atomics.
- The smoothing sum divides by `V` first (`inv_v`).

A host-only emulation of `lq_orthonormalize` checked the logic. The function
text was cut verbatim from the `.cu` file and built with `g++` (32 threads for
the lanes, shuffles emulated). This is not the HIP compiler and it does not show
anything about the GPU.

- 41 cases (dense, tiny, huge, mixed scales, exact multiples, noise columns,
  NaN, inf, subnormal, zero) gave 0 violations of the contract: every column is
  zero or unit, all pairs are orthogonal to about 3e-7, all values are finite.
- A NaN column and an inf column became dead, as in the oracle.
- For independent columns the emulation and the oracle differ by at most 3e-8.
- Dependent noise columns can differ, because their direction depends on the
  summation order. Both satisfy the contract.

[INFERENCE] The extra work in the fold is about one extra wave-0 pass per call.
It was not measured.

### Limits

- A state with a NaN input gives garbage in FP32 too. [INFERENCE] HIP `fmaxf`
  and `fminf` ignore NaN, so the kernel may turn a NaN state into finite garbage
  in the INT8 section. This did not change.
- For large states the products of the power iteration (`S X`, `S S^T Q`)
  overflow. The guard makes the affected columns dead, so the INT8 residual
  carries the state alone (Table 2).

### Tests

- CPU, `test_gdn_leapquant_oracle.py`: zero, tiny (1e-8 to 1e-42), huge (1e6 to
  1e37), the float limit, exactly low-rank (ranks 1 to 4, 3 seeds, 4 scales),
  integer low-rank, rank above 4, six dynamic-range states, neighbour heads,
  `_orthonormalize` unit tests and the rank-1 trajectory. 92 tests.
- GPU, `test_gdn_leapquant_safety_rdna2.py`: the same state list through the
  `store` and `materialize` kernels, a neighbour-head check, and the rank-1
  trajectory with window flushes. 3 tests. Pending GPU validation.

## 2. Determinism under load

The RFC reports a ROCm fold and replay that differed from run to run in the
prototype. It asks for a test that runs the same input 200 times on a loaded
GPU.

Review of `gdn_leapquant_rdna2.cu`: the file has one atomic, the shared-memory
`atomicOr` of a constant in `lq_fold`. Its result does not depend on the order.
All other reductions use shuffles and shared-memory sums in a fixed order
(`lq_ksum`, `lq_wsum`, `lq_reduce_over_v`). A comment at lines 94 to 99 of the
file says the same. This does not replace the test.

GPU tests (pending GPU validation), 200 repeats each (`N_REPEATS`):

- `test_store_normalize_decode_are_bitwise_reproducible_under_load`: cold fold
  (`store`), warm fold (`normalize`), replay and decode on identical inputs.
  Outputs, states and the whole blob must be bitwise equal in every repeat. A
  second stream runs FP16 matmuls at the same time. The output buffers hold a
  different poison value in each repeat, so an element that a kernel does not
  write shows.
- `test_mono_gdn_lq_is_bitwise_reproducible_under_load`: the same for the
  MonoKernel LeapQuant path, with the scratch buffer reused.

**Status: not yet run.** The tests are written and skip without a gfx103x GPU.
CPU-verified: nothing. A CPU run cannot show a GPU race.

## 3. Stale state of recycled blocks

A recycled cache block holds the blob of an earlier request. The RFC asks that
scales, smoothing factors, compensators, records and cursor are overwritten
before the first read of the new request.

### What the serving path does with a recycled block

- `KVBlockZeroer` skips Mamba layers (`vllm/v1/worker/utils.py`, line 145). A
  recycled block is not zeroed.
- The attention metadata builder classifies a first chunk as a prefill, to mask
  recycled state (`vllm/v1/attention/backends/gdn_attn.py`, lines 510 to 531).
- The prefill path reads the leftovers once. It runs `normalize` and
  `materialize` on the block and then sets the initial state of every row
  without an initial state to zero
  (`qwen_gdn_linear_attn.py`, lines 2721 to 2745). The chunk kernel never sees
  the leftovers. The pass can fold them, which writes to the block. So both
  kernels must survive garbage: they read a header outside the range as empty
  and they index no row outside the ring.
- The prefill ends with `store` (lines 2817 to 2821). The store writes `z`,
  `bscale`, `csmooth`, `comp_k`, `comp_u` and the header words `[0:2]`. It
  writes no record. The cursor is 0, so the old records are dead.

### Findings

- No stale read reaches an output when the serving contract holds. The state is
  a function of `z`, `bscale`, `csmooth`, the compensators, the first `cursor`
  records and the header. The prefill pass reads leftovers and drops them. The
  tests with `prefill_pass` check that nothing from that pass survives the
  store.
- The first PR tests start from random bytes in the destination of the store
  test and of the pre-copy test. The store test checks a tolerance. The new
  tests add: a bitwise comparison with a clean block through decode, flush and
  later reads, with and without the prefill pass; a valid-looking stale header
  with stale records; NaN in every section; rows past the cursor; a fixed list
  of headers outside the range; and the MonoKernel on a zero block and after a
  store.
- The old oracle did not read an invalid header as empty. It raised `IndexError`
  for `cursor` above the ring and `AssertionError` for a negative `pending`. The
  kernels read such headers as empty (`lq_header_ok`, `gdn_leapquant_layout.cuh`
  lines 63 to 66). The oracle now does the same (`_sane_header`). This is a fix
  of the oracle, not of a kernel.
- The align pre-copy reads the header of the source block without a range check
  (`v1_mamba_rdna2.cu`, lines 233 to 256). A source header that fails the check
  can give a destination header that passes it. Example: cursor 32 and pending
  34 (window 32, capacity 34) give cursor 33, so the copy would replay 33
  records that the source reads as empty. The source is a live home block, and
  its header always passes the check, because it comes from `store`,
  `normalize` or decode. The kernel does not know the capacity, so it cannot
  check. Not changed. The new tests cover the leftovers of the destination.
- Decode and the MonoKernel read `pending` through `lq_header_ok`, and tile 0
  writes `pending` at the end (see section 4). This is the one case where a
  stale header can give different tiles different cursors. It needs a header
  with a valid cursor and an invalid `pending`, and a launch that skips
  `normalize`. The serving path always runs `normalize` first, and it rewrites
  the header. Not changed. A one-line hardening would be to test only the cursor
  in decode: `lq_header_ok(hdr[0], 0, L)`. That also needs an oracle change.

None of these needed an `xfail`. No test reproduces a stale read on the CPU
oracle.

### Tests

CPU, `test_gdn_leapquant_oracle.py` (30 tests):

- `test_store_overwrites_everything_a_later_read_depends_on[kind, prefill_pass]`
- `test_first_decode_after_store_never_replays_leftover_records[kind, prefill_pass]`
- `test_fresh_zero_block_is_the_zero_state`
- `test_records_beyond_cursor_are_never_read[nan, inf, huge]`
- `test_out_of_range_header_reads_as_empty[9 headers]`
- `test_copy_slot_hands_out_only_its_own_blob`

The leftover kinds `kind` are random bytes, the blob of a previous tenant, the
same with NaN in every section, and a valid-looking header with stale records.
`prefill_pass` adds the `normalize` and `materialize` that the serving prefill
path runs on the leftovers before the store.

GPU, `test_gdn_leapquant_safety_rdna2.py` (31 tests, pending GPU validation):

- `test_new_tenant_is_independent_of_the_leftovers_of_a_recycled_block[kind, prefill_pass]`:
  store, first decode, flush and later reads equal those of a clean block, bit
  for bit.
- `test_first_decode_on_a_fresh_zero_block_without_a_store` and
  `test_mono_gdn_lq_on_a_fresh_zero_block_matches_fp32_mono`.
- `test_mono_gdn_lq_after_a_store_is_independent_of_leftovers[kind]`.
- `test_header_that_fails_the_range_check_reads_as_empty[10 headers]`: the
  record rows of the dirty block hold NaN, so any read of them shows.
- `test_records_past_the_cursor_are_never_read[nan, inf, 6e4]`.
- `test_align_precopy_overwrites_the_leftovers_of_its_destination[kind]`: the
  copy must equal the oracle copy in every byte.

## 4. Flush race

The RFC reports a race in the prototype. At flush time, several programs of one
launch read and wrote the same record row.

### Race by construction

Line numbers are for `csrc/rocm/gdn_leapquant_rdna2.cu` unless noted. They are
for the base commit `7b4aff7fe3` plus the commits of this change. Other changes
can move them. The function names stay.

**Normalize** (`gdn_lq_normalize_kernel`, lines 416 to 460):

- One workgroup owns one blob: (request, head), and the layer when
  `layer_offsets` is set (`blockIdx.y`, line 430). A grouped launch over layers
  uses a different base address for each layer.
- Thread 0 reads the header (433 to 446). A barrier follows (447). The early
  exit when no fold is needed (449 to 455) is uniform for the workgroup.
- All threads read every section that the fold needs into registers
  (`lq_load_full`, line 457). A barrier follows (458). Only then `lq_fold`
  writes.
- In `lq_fold`, the old `comp_k` is read first (258 to 273). The new `comp_k`
  and `comp_u` are written after the barriers that end `lq_apply_s`,
  `lq_apply_st` and `lq_orthonormalize`. `csmooth`, `bscale`, `z` and the header
  are written last (323 to 363).
- The fold never writes a record. No program of the launch writes a row that
  another program reads.

**Decode** (`gdn_lq_decode_kernel`, lines 540 to 677):

- One workgroup per V tile of 32 rows and per (request, head). Each reads the
  records `t < c` (lines 606 to 612). `rec_u` is read in its own columns.
- Each writes records at rows `c` and above. `rec_u` is written in its own
  columns (line 667). `rec_k` and `rec_g` are written by tile 0 only
  (`i_v == 0 && v_local == 0`, lines 668 to 673). `pending` is written by
  tile 0, lane 0 (line 676). `hdr[0]` is never written.
- Rows read (below `c`) and rows written (`c` and above) do not overlap. Columns
  written by different tiles do not overlap. The eight lanes of one row write the
  same `rec_u` value to the same address. The value is the same in all lanes,
  because the xor-shuffle sums give all lanes the same bits.

**MonoKernel** (`mono_gdn_rdna2.cu`): same structure. The header is read at lines
290 to 292, the records are written at 391 to 403, and `pending` is written by
tile 0 at 433 to 437. The comment at line 434 says that other tiles read only
the cursor word. That is not exact: they also read `pending` through
`lq_header_ok`. The result is the same as for decode.

**One case of disagreement.** All tiles of a head read the header at their start.
Tile 0 writes `pending` at its end. For a normalized header, `pending` is 0
before and `min(T, cap - c)` after. `lq_header_ok` is true in both cases, so every
tile gets the same cursor. For a header that is not normalized, and has a valid
cursor and an invalid `pending`, a tile that starts after tile 0 ends can read a
valid header and a different cursor. See section 3.

**Preconditions that the kernels cannot check:**

- Two live requests do not share a state index in one launch.
- The regions of `layer_offsets` do not overlap. The page packing work must keep
  this.
- No other stream uses the same blob at the same time.
- A padded row has an index `<= 0`. Normalize returns early for it (line 429).
  Decode writes zeros to its output and returns (lines 570 to 575). Neither
  changes a blob.

### Tests

GPU, `test_gdn_leapquant_safety_rdna2.py` (10 tests, pending GPU validation):

- `test_grouped_flush_of_full_windows_equals_per_layer_launches[window, mixed, load]`:
  32 requests and 6 layers reach the window in the same launch with MTP
  accepted counts (0 to 3). One grouped normalize over all layers must equal,
  bit for bit, one launch per layer. 25 repeats (`GROUPED_REPEATS`). With and
  without a second-stream load. Windows 16 and 32.
- `test_batched_decode_equals_singleton_launches[window]`: a decode over many
  requests must give each request what a launch over that request alone gives.

These tests can show a race. They cannot show that none exists. The review above
is by reading the code and is not a proof.

## 5. End-to-end decode benchmark

The RFC asks to extend `benchmarks/replayssm/e2e_decode_speedup.py` to compare
FP32, ReplaySSM FP32 and ReplaySSM FP16 or BF16.

Added:

- Arm `leapquant`. It sets `VLLM_RDNA_GDN_LEAPQUANT=1` in its own worker
  process, imports the LeapQuant module only in that process, and fails off
  gfx10x. Every other arm refuses to start when the flag is set in the
  environment.
- `--modes` (default `standard,replayssm`, unchanged). The first mode is the
  baseline for the speedup lines.
- `--num-speculative-tokens N` for MTP, `--monokernel`, `--tensor-parallel-size`,
  repeatable `--llm-kwarg KEY=JSON` and `--env KEY=VALUE`.
- `--dry-run`: prints the plan and the worker commands. It imports neither
  `torch` nor `vllm`.

The arms that existed before have the same engine arguments, environment and
report text. A test pins this.

FP32 baseline against LeapQuant on the RDNA path:

```bash
python benchmarks/replayssm/e2e_decode_speedup.py --model-id <GDN model> \
    --dtype float16 --modes standard,leapquant --mamba-ssm-cache-dtype float32 \
    --num-speculative-tokens 2 --monokernel --no-disable-flashinfer-autotune \
    --env VLLM_USE_V2_MODEL_RUNNER=1 --env VLLM_GDN_HIP_PREFILL=1 \
    --batch-size 8 --num-steps 512 --warmup-steps 64
python benchmarks/replayssm/e2e_decode_speedup.py --modes standard,leapquant --dry-run
```

**Not done: the 16-bit per-window baseline.** It needs a Gated DeltaNet ReplaySSM
decode kernel
([vllm#62004](https://github.com/vllm-project/vllm/pull/62004)). This tree does
not have it, and `use_replayssm` is for Mamba2 only. A new kernel is outside this
work. The script states this in its docstring.

CPU-verified: `tests/benchmarks/test_replayssm_e2e_decode_speedup.py` (17
tests): the arguments, the arm and environment rules, the default arms, the
worker commands, `--help` and `--dry-run` without `torch`. No benchmark ran.

## How to run

CPU (any host):

```bash
pytest -v -s tests/kernels/mamba/test_gdn_leapquant_oracle.py
pytest -v -s tests/benchmarks/test_replayssm_e2e_decode_speedup.py
```

GPU (gfx103x only; the tests skip elsewhere):

```bash
pytest -v -s tests/kernels/mamba/test_gdn_leapquant_rdna2.py
pytest -v -s tests/kernels/mamba/test_gdn_leapquant_safety_rdna2.py
```

On a CPU host the GPU tests skip: 68 skipped for the two files together (46 new
tests and 22 from the first PR).

The author also ran the new GPU test file once on a CPU box. The FP32 oracle
stood in for the HIP ops and the MonoKernel ops were stubs: 46 passed. This
checks the test code (shapes, helpers, comparisons) only. The stand-in is not in
the tree. It says nothing about the kernels.

## Pending GPU checklist

Nothing below was done. Each step needs a GPU or a build.

- [ ] Build `_rocm_C` with the changed `gdn_leapquant_rdna2.cu`. The build uses
      `-Werror=unused-variable`.
- [ ] Run `test_gdn_leapquant_rdna2.py`. The fold changed for every blob, so the
      first PR tests must pass again.
- [ ] Run `test_gdn_leapquant_safety_rdna2.py` (46 tests). The tests never ran
      on a GPU. A failure is a kernel bug or a test bug. Find out which one.
- [ ] Run the determinism tests (200 repeats) on a GPU that has other work. Run
      them more than once.
- [ ] Run the grouped-flush and batched-decode tests on the target GPU.
- [ ] Re-run the quality checks of the first PR (decode-path KL, gated checks).
      The new fold gives other compensators for rank-deficient heads.
- [ ] Measure the fold time before and after. [INFERENCE] The change is small.
- [ ] Remove the "NOT BUILT, NOT RUN" note in `gdn_leapquant_rdna2.cu`.
- [ ] Run the benchmark arms on the target GPU and record the numbers here.
- [ ] Confirm that the smoothing factor of a normal state is bit-identical to
      the first PR. The CPU check compared `mean()` with divide-then-sum in
      PyTorch only.
- [ ] Decide on the decode hardening in section 3 (`lq_header_ok(hdr[0], 0, L)`)
      and on a capacity check in the align pre-copy.

## Not done

- GPU validation of every item above. No GPU was available.
- The 16-bit per-window baseline arm (section 5).
- A new kernel or a change to the decode and MonoKernel header logic.
- Config flags, startup refusals, the format identifier and page packing. They
  belong to other work.
