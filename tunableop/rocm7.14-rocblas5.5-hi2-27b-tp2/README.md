# rocm7.14-rocblas5.5-hi2-27b-tp2

Opt-in derived profile for **Qwen3.8-27B-HI2-3.0bpw at TP=2** (recipe `27b-hi2-tp2`):
the canonical `rocm7.14-rocblas5.5` rank files 0-3 byte-for-byte (3330 rows) plus
**36 new rows per rank = 3366**, identical on every rank (a TP=2 worker on device
ordinal `r` reads `tunableop_results<r>.csv`; ranks 0 and 1 serve, 2 and 3 exist for the
helper's four-file check). Same rocBLAS build as the canonical profile
(`5.5.0.cd957402`, `librocblas.so.5` sha256 `f30bb442e9b5...`).

* **Selection.** Registered with `"auto": false` in `tunableop/profiles.json`: auto-select
  never picks it. `tools/rdna/recipes/27b-hi2-tp2.env` sets
  `TUNABLEOP_PREFER=rocm7.14-rocblas5.5-hi2-27b-tp2`; no other recipe does, so their rows
  stay byte-identical. `TUNABLEOP_PREFER=` (empty) or
  `TUNABLEOP_PROFILE=rocm7.14-rocblas5.5` is the rows-off arm of the same tree. The boot
  log prints `TunableOp profile ... preferred (TUNABLEOP_PREFER) ...` and
  `TunableOp rows per rank ...: 0=3366 1=3366`.
* **Single row set instead?** `tools/rdna/prof/tunableop_missing.py harvest --profile
  rocm7.14-rocblas5.5 --tuned <the 36 rows>` merges them additively into the canonical
  profile (the keys do not collide with any canonical key).

## Rows added (per rank)

| slot | per-rank R x K | M | rows |
| --- | --- | --- | ---: |
| `gate_up_proj` | 17408 x 5120 | 1023, 1024, 2046, 2047, 2048 | 5 |
| `in_proj_qkvz` | 8192 x 5120 | 1023, 1024, 2046, 2047, 2048 | 5 |
| `qkv_proj` | 7168 x 5120 | 1023, 1024, 2046, 2047, 2048 | 5 |
| `out_proj` / `o_proj` | 5120 x 3072 | 1024, 2046, 2047, 2048 | 4 |
| `in_proj_ba` | 48 x 5120 | 1023..1031, 1033, 1036, 1039, 1042, 1045, 2046, 2047, 2048 | 17 |

Keys are `GemmTunableOp_Half_TN,tn_{R}_{M}_{K}_ld_{K}_{K}_{R}`. `kernel_table.csv` holds the
measured heuristic and tuned time and TF/s of every candidate (adopted or not).

## Not adopted

* `gate_up` and `out`/`o` at M = 1025..1031 and 1033..1045, `qkvz` / `qkv` at 1025..1031 and
  1033..1045: the tuner returned the `Default` solver or a solver less than 3 % faster
  than the heuristic (median of four alternating passes). An odd M one to a few rows past
  a tile multiple costs a whole extra tile row, and no rocBLAS solution avoids that.
* `down_proj` (5120 x 8704) and the MTP `fc` (2560 x 10240) at every census M: the
  heuristic is as fast (-1.3 % .. +0.5 %).
* Arbitrary chunk tails (probe 834 / 1015 / 1018 / 172, warm repeat 475 / 27, prompt
  logprobs 903 / 906 / 297), the unfused prefill attention GEMMs (`s_len` varies per chunk),
  the MTP drafter's dense layers at M = 9..24 and the LM head rows (cap 32 keeps the GEMV
  for <= 32 rows): see `docs/rdna2/hi2-27b-tunableop-2026-11-15.md`.

## How the rows were made and checked

`tools/rdna/rdna_hint_flex/tunableop_hi2.sh` on HIP 4 (V620, bus 0xdf), one shard per
`flock` hold: offline tuner (TunableOp defaults: 30 ms and at most 100 iterations per
candidate, hipBLASLt off) -> cold-weight CUDA-graph timing of the heuristic and the rows
(median of 4 passes for the borderline keys) -> `rowcheck.py` (tuned vs rocBLAS default vs
fp64: 89 rows, 0 rejected, worst error ratio 1.00) -> curate (adopt >= 3 %) -> harvest into
this folder. `tunehit.py`: all 31 stage-one rows found at lookup on the box tree.
`provenance.json` has the environment and the key list.

## Gate status

Single-GPU model-level check (real checkpoint, TP=2 GEMMs emulated at TP=1, 3 prompts x
64 greedy tokens): tokens identical to the rows-off run, max |dlogit| 0.094 over the top-20
raw fp16 logits. **Not yet run: the deterministic concurrency probe on the TP=2 pair**
(HIP 2,3); the row set is not accepted until it passes (commands in the doc, section 7).
