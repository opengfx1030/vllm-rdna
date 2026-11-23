# Inkling-2 pipelined prefill attention tile: qualification and archdiff receipt

Evidence for `i2_tile_pipe_kernel` (`tile_impl=1` of the `inkling2_attention` op), which
computes exactly what the shipped `i2_tile_kernel` (`tile_impl=0`) computes. Kernel and
timings: `inkling2-attn-kernels.md` section 12. This page is the correctness side: a
depth battery against an fp64 oracle with a bit-for-bit comparison of the two kernels,
hand mutants the battery must kill, and the six-arch code-generation diff.

## Revisions

- BASE: `9285bf8523` (`origin/rdna_extras`).
- HEAD: BASE plus the prefill tile change. Files that were run:
  `csrc/rocm/inkling2_attn_rdna2.cu` (sha256 `6e095d88064c465f...`, BASE
  `bfc36efb2d9cf75e...`), `csrc/rocm/ops.h` (`cd3be215b9f9747d...`),
  `csrc/rocm/torch_bindings.cpp` (`02f4c2c93ab3e2bf...`), and `vllm/_custom_ops.py` with
  the `tile_impl` keyword (the wrapper policy: a sliding layer of at most 32 query
  tokens keeps `tile_impl=0`). These are the blobs of `0cd91cb62c`.
- Box: `par1-cs25`, HIP 2 (gfx1030, V620), the ROCm 7.14 venv of the box.

## The battery

`tests/kernels/core/test_inkling2_attn_prefill_depth_rdna2.py` (tooling in
`tools/rdna/inkling2/attn_jit.py` and `attn_tile_mutants.py`).

- **Oracle.** fp64 on the GPU, in blocks of keys; the same arithmetic as the oracle of
  `tests/kernels/attention/test_inkling2_attn_rdna2.py` (a CPU test pins them against
  each other). A random world at 100k keys averages the values and every output is about
  1e-2 whatever the kernel does, so the worlds are *needle* worlds: a few keys per
  request (the first / last visible position, both sides of the window and of the
  relative-bias reach of the first and last query row, page and tile edges, key 0, key
  L-1, random depths) are aligned with the queries and score about 17 logits above the
  bulk. A key that is wrongly visible, masked, read from the wrong slot or given the
  wrong bias moves an output by 30 - 610 times the tolerance
  (`test_needle_worlds_are_sensitive`, measured on CPU at q = 33 / 257 / 1024).
- **Poisoned cache.** Unreferenced pages, the slots past the sequence end, pages outside
  every window and the keys below the window of the first row (inside its page) are NaN.
  A read of any of them shows in the output (`max |out|` is checked finite, and the
  workspace and the output are NaN-poisoned before every call).
- **Tolerance.** `|err| <= 1e-4 + 2^-7 * sum_j p_j |v_j| / den`: the fp16 probabilities
  carry a relative error of 2^-11 each, so the error of an output follows the magnitude
  of the sum, not its value. Calibrated on the shipped kernel: worst error 3.6e-3 of the
  magnitude (worst `err / bound` 0.44, i.e. 2.3x headroom), over 288 error checks.
- **Bit identity.** The variant (`tile_impl=1`) must give the same bits as the shipped
  kernel (`tile_impl=0`): the output tensor and the split-K workspace, NaN payloads
  included, and both must be inside the oracle bound.
- **Coverage.** Query chunks of 1024 / 257 / 33 / 2 / 1 rows (fresh and mid-sequence)
  at L = 1025 ... 144001 (page-, tile- and window-boundary lengths 16383 / 16384 / 16385,
  100353, 131073, 144001), sliding window 1024 and full attention, GQA 24 / 8 (`g` 1 - 4),
  sinks, relative bias with reach edges, no-gate strided inputs, `splits_tile` 1 / 2 / 5 /
  16 / auto, mixed decode + prefill batches, cache block sizes 16 / 32 / 64 / 128 (the
  fast staging needs 64 or more), the outermost layers, `(L - q) % 16` swept over all 16
  values for q = 256 / 300 (a 16-row block starts at every position mod 64), and the
  wrapper default (`tile_impl=None`) against `tile_impl=0`.

Run (CPU tests: `HIP_VISIBLE_DEVICES= I2_ATTN_ORACLE_DEVICE=cpu`):

```bash
# the shipped op alone (the repo .cu, tile_impl 0, JIT-built): oracle checks only
python -m pytest tests/kernels/core/test_inkling2_attn_prefill_depth_rdna2.py -q
# the variant against the shipped op: needs the op with tile_impl (installed _rocm_C of
# this tree, or I2_ATTN_FORCE_JIT=1 to JIT-build the repo .cu)
I2_ATTN_FORCE_JIT=1 I2_ATTN_REQUIRE_VARIANT=1 \
python -m pytest tests/kernels/core/test_inkling2_attn_prefill_depth_rdna2.py -q
```

### Result

| run | tests | outcome |
|---|---|---|
| BASE `.cu` (24-argument op, no variant), HIP 2 | 306 | 297 passed, 9 skipped (variant tests), 421 s |
| HEAD `.cu` + wrapper, `I2_ATTN_FORCE_JIT=1 I2_ATTN_REQUIRE_VARIANT=1`, HIP 2 | 306 | **306 passed, 0 skipped**, 383 s |

In the second run the variant is bit-identical to the shipped kernel in every case
(outputs and workspace), and its error against the oracle is the shipped kernel's:
288 checks each, max abs error 3.588e-3, max error / magnitude 3.556e-3, worst
`err / bound` 0.441, for both kernels.

## Hand mutants

`python tools/rdna/inkling2/attn_tile_mutants.py --phase build` (CPU, JIT-compiles the
unmutated kernel and every mutant), then `--phase test` (HIP 2: the battery once per
mutant, the variant = the mutant, the shipped reference = `tile_impl=0` of the unmutated
file; `-x`, first failure kills). Run with `--source csrc/rocm/inkling2_attn_rdna2.cu
--kernel i2_tile_pipe_kernel --variant-mutants --max-l 20000`: exit 0, the unmutated
control passes, 22 of 22 mutants are killed (and one probe, below, survives).

| mutant | edit in `i2_tile_pipe_kernel` | killed by |
|---|---|---|
| `window_edge` | sliding window: the key one before the window is visible | `test_prefill_depth[sliding-q1024-L1025]` |
| `window_edge_in` | sliding window: the oldest key of the window is masked | same |
| `causal_edge` | causal mask: a row sees the key after it | same |
| `bias_reach_in` | relative bias: the key at distance `rel_extent` gets a bias | `test_prefill_depth[full-q1024-L1025]` |
| `bias_reach_out` | relative bias: the key at distance `rel_extent - 1` gets none | `test_prefill_depth[sliding-q1024-L1025]` |
| `bias_clamp_self` | relative bias: the key at distance 0 reads proj row 1 instead of row 0 | same |
| `v_pair_swapped` | V staging: the pair (m, m + 32) is stored swapped | same |
| `split_one_tile` | split-K: a split walks one tile too few (`i2_tile_split`) | same |
| `window_start_tile` | key range: the first tile is one too late (`i2_tile_split`) | `test_prefill_depth[sliding-q257-L16383]` |
| `skip_bias_one_early` | bias skip: the tile whose last key is at distance `rel_extent - 1` is skipped | `test_prefill_depth[sliding-q257-L16383]` |
| `skip_bias_last_row` | bias skip: tested at the last row of the CTA instead of the first | `test_prefill_depth[sliding-q1024-L1088]` |
| `mask_skip_one_late` | mask skip: an interior tile may end one key past the first row of the block (survived a 43-world battery of the kernel's author: it differs only when a full 16-row block starts at 62 mod 64) | `test_prefill_depth[sliding-q257-L16383]` |
| `page_parity` | page table double buffer: the next tile's pages land in the current slot | crash of the process (`Fatal Python error`) in the 19th test; the run is not green |
| `prefetch_same_tile` | K prefetch: the next tile's K rows are loaded from the current tile | `test_prefill_depth[sliding-q1024-L1025]` |
| `prefetch_v_same_tile` | V prefetch: the next tile's V rows are loaded from the current tile | same |
| `fast_path_hi` | V fast path: taken when the tile runs one key past the last row | `test_prefill_depth[sliding-q1024-L16383]` (V of the NaN slot past the end) |
| `fast_path_lo` | V fast path: taken when the tile starts one key before the window | `test_prefill_depth[sliding-q1024-L16384]` (V of the NaN key below the window) |
| `row_share_lane` | probabilities in registers: the DPP `row_share` reads the next lane | `test_prefill_depth[sliding-q1024-L1025]` |
| `row_share_word` | probabilities in registers: the DPP `row_share` reads the other pair word | same |
| `pack_swapped` | probabilities in registers: the pair `(p0, p2)` is packed swapped | same |
| `pack_swapped_edge` | the same on the edge-tile path | same |
| `v_buffer_early` | V rows stored before every QK reader is done (the barrier `(2)` removed: a race) | `test_prefill_depth[full-q1024-L4999]` |

Not counted:

- `prefetch_guard` (drops the `tile + 1 < t1` guard of the last-tile prefetch) is a
  *probe*: the extra load is never used, so the output cannot show it unless it faults.
  It survives, as expected. The guard is covered by reading the code, not by a test.
- Equivalent edits, documented in `EQUIVALENT` of the tool and not run: the K twins of
  `fast_path_hi` / `fast_path_lo` (the extra key is masked by the causal / window
  select, a NaN score never reaches the sum), `bias_idx_clamp_high` (only the proj row
  of a key that `inb` zeroes), `bias_clamp_low` (only keys after the row, which the
  causal mask removes).
- `fast_path_lo` and `fast_path_hi` were first reported as holes: with a cache whose
  slots below the window were finite data (the real content of a partially evicted
  page) the edit is invisible. The battery now poisons them (see above): the shipped
  kernel reads none of those keys, so the check is within the contract of both kernels.

## Archdiff receipt (six arches)

Code-generation diff of the csrc change, BASE `9285bf8523` against HEAD (BASE with the
three csrc files above), `tools/rdna/archdiff.sh` in tree-directory mode on `git archive`
exports, CPU only (`HIP_VISIBLE_DEVICES=` empty), `nice -n 19`, `JOBS=8`,
`~/w4a8_runs/audit/build.lock`, `KEEP_BUILD=1`.

```bash
D=$HOME/w4a8_runs/inkling2-work/Inkling2Archdiff
cd $D/src-final
BASE=$D/src-9285 HEAD=$D/src-final WORK=$D/work-final2 JOBS=8 KEEP_BUILD=1 \
EXPECT='i2_tile_pipe|i2_row_share|tile_pipe' \
MAY_DIFFER='gfx90a gfx942 gfx950 gfx1030 gfx1100 gfx1200' \
HIP_VISIBLE_DEVICES= nice -n 19 bash tools/rdna/archdiff.sh
```

Exit 0: `PASS: 4 differing function(s), all expected by EXPECT / MAY_DIFFER`
(`archdiff.json` sha256 `115eb7b98c10ba5d...`). Allowed `__LINE__` shifts (derived from
the source diff): -2 +2 +21 +555 +557 +559 +562.

| arch | objects | functions base / head | identical | changed | only in base | only in head | verdict |
|---|---|---|---|---|---|---|---|
| gfx90a | 6 | 121 / 121 | 121 | 0 | 0 | 0 | no difference |
| gfx942 | 6 | 121 / 121 | 121 | 0 | 0 | 0 | no difference |
| gfx950 | 6 | 121 / 121 | 121 | 0 | 0 | 0 | no difference |
| gfx1030 | 6 | 151 / 155 | 151 | 0 | 0 | 4 | expected only |
| gfx1100 | 6 | 121 / 121 | 121 | 0 | 0 | 0 | no difference |
| gfx1200 | 6 | 121 / 121 | 121 | 0 | 0 | 0 | no difference |

The only differences are the four instantiations of the new kernel on gfx1030, under the
existing RDNA2 guard; every function that exists in BASE (121 on each arch, 151 on
gfx1030, including `i2_tile_kernel<1..4>`) is bit-identical in the disassembly:

| gfx1030 function | instructions |
|---|---|
| `i2_tile_pipe_kernel<1>(I2AttnArgs)` | 4072 |
| `i2_tile_pipe_kernel<2>(I2AttnArgs)` | 5818 |
| `i2_tile_pipe_kernel<3>(I2AttnArgs)` | 7482 |
| `i2_tile_pipe_kernel<4>(I2AttnArgs)` | 9280 |

The other five arches get no function from the new kernel at all (nothing is emitted
outside the guard). An earlier run on a work-in-progress `ops.h` (only formatting differed
from the final one) gave the same table.

## Limits

- The qualification is as strong as the worlds: finite inputs (the interior-tile mask
  skip differs from the shipped kernel only for NaN / inf scores), gfx1030 only,
  `q_len >= 2` only (a single-token query never reaches the tile kernel), cache block
  sizes below 64 take the shipped per-key path (covered by `test_prefill_block_sizes`).
- The oracle bound is statistical (2^-7 of the magnitude); the bit comparison with the
  shipped kernel is the exact check.
- The whole battery ran on one GPU (V620, HIP 2), eager, `-O2` JIT builds of the
  repo file; the CMake build of `_rocm_C` was not run (the archdiff built it for the
  six arches, including the gfx1030 objects).
