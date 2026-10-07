# PR #34 validation — FA-RDNA2 D=128 prefill (gfx1030), 2026-10-02

PR: opengfx1030/vllm-rdna **#34** `perf(rocm): FA-RDNA2 D=128 prefill through the
register-O GQA kernel` (draft, head `claude/fa-rdna2-prefill-d128`).
Base: `rdna_extras` @ `bc5fbee5d`.

## STEP 0 — head_dim determination

Flash-Next `config.json` `text_config.head_dim = **256**`, `num_attention_heads=24`,
`num_key_value_heads=2` → **GQA group = 12 (even)**.

⇒ The D=256 even-group path is exactly the one Flash-Next already runs, so this PR
is **bitwise-identical (a no-op) for Flash-Next by design**. The run is a
*no-regression proof*; the PR's D=128 value is for Qwen3/Llama-class users.

## Result: MERGED

`284fbad0a` on `rdna_extras` (pushed; `origin/rdna_extras` = `284fbad0a`).
Re-authored to `BlivionIaG <kev29lt@gmail.com>` per fork policy (PR head commit
was agent-authored); tree for the 5 PR files is byte-identical to the PR head.

## Build / ISA

| gate | baseline | PR#34 |
|---|---|---|
| `ninja _rocm_C` | BUILD_EXIT=0 (`_rocm_C.baseline.so`) | BUILD_EXIT=0 (`_rocm_C.patched.so`, +216 KB) |
| instantiations | `gqa_kernel_256<2,8>` only | `<128,2,8> <128,1,16> <256,2,8> <256,1,16>` |
| `<128,2,8>` | — | 80 VGPR / 46 SGPR, 0 spills, 0 scratch |
| `<128,1,16>` | — | 80 VGPR / 48 SGPR, 0 spills, 0 scratch |
| `<256,2,8>` | — | 93 VGPR / 51 SGPR, 0 spills, 0 scratch |
| `<256,1,16>` | — | 93 VGPR / 51 SGPR, 0 spills, 0 scratch |

`--offload-arch=gfx1030`, `max_flat_workgroup_size=256` (8×wave32), no warnings.
(Different compiler than the PR's offline clang 18, hence different VGPR counts;
0 spills in all four.)

## Correctness

- **72/72** `pytest tests/kernels/attention/test_fa_rdna2_{writer_layout,shape_sweep}.py`
  pass on the patched build (covers D=128 G=4/7/1, D=256 G=6/3, dense + interleaved).
- **D=256 even group bitwise identical**: cross-build probe, Flash-Next shape
  (D=256, 24/2, ~1.5k tokens) → `torch.equal == True`, `max_abs_diff = 0.0`.
  SASS of `<256,2,8>` differs by exactly 2 algebraically-equivalent lane-id
  instructions (`(v0&31)<<4` vs `(v0<<4)&0x1f0`).

## Flash-Next matrix (D=256, TP=4, FULL_AND_PIECEWISE, prefix-caching, FA-RDNA2)

8 cells × 2 arms. `out` = aggregate output tok/s; `TTFT` ms. `c8_16384` is 8
prompts queued (two waves).

| cell | baseline out | patched out | Δ | base TTFT | pat TTFT |
|---|---:|---:|---:|---:|---:|
| MTP0 c1 1k/512 | 41.18 | 41.16 | −0.05% | 622.5 | 622.8 |
| MTP0 c8 1k/512 | 166.38 | 173.70 | +4.4% | 3592.5 | 2772.8 |
| MTP0 c1 16k/1k | 30.56 | 30.67 | +0.4% | 9520.8 | 9463.1 |
| MTP0 c8 16k/1k | 69.50 | 70.01 | +0.7% | 45014.6 | 44267.6 |
| MTP2 c1 1k/512 | 67.22 | 65.77 | −2.2% | 603.1 | 626.9 |
| MTP2 c8 1k/512 | 128.41 | 130.14 | +1.3% | 5369.9 | 2919.2 |
| MTP2 c1 16k/1k | 35.90 | 47.49 | +32%* | 9800.2 | 9863.7 |
| MTP2 c8 16k/1k | 67.13 | 67.95 | +1.2% | 48036.3 | 47018.9 |

Full per-cell metrics (TTFT, TPOT, per-req decode, ITL p50/p99, MTP acceptance) in
`matrix_combined.csv`. `*` MTP2 c1 16k is spec-acceptance variance (baseline
accept 47.2% vs patched 95.7%), not a kernel effect — the kernel is bitwise
identical. All coherence 2/3 with `garbage=ok` (the 2/3 is a benign
literal-substring miss on the "gpu" probe; no empty/repeat/bang/nonprintable
garbage in any arm).

**No regression** on any cell beyond run-to-run noise.

## Routing toggle A/B (patched, `VLLM_FA_RDNA2_GQA_MODE` subgroup vs off)

| cell | subgroup | off | Δ |
|---|---:|---:|---:|
| c1 1k/512 | 41.16 | 41.24 | +0.2% |
| c8 1k/512 | 173.70 | 171.77 | −1.1% |
| c1 16k/1k | 30.67 | 30.71 | +0.1% |
| c8 16k/1k | 70.01 | 69.54 | −0.7% |

Identical — as expected for D=256 even groups (both modes resolve to the same op).

## D=128 kernel A/B (the PR's actual target) — `benchmarks/kernels/benchmark_fa_rdna2_prefill.py`

Median µs, cudagraph timing, patched build, **clean re-run on GPU 5** (a
co-tenant job (`tf-measure-cells.py`) started on GPU 4 at ~01:57, after the
Flash-Next matrix finished; the first benchmark pass overlapped it and the
clean re-run is within 1–2% — see `remote/bench_clean_gpu5.out`).
**gqa = new path; short/varlen/splitk = old path.**

D=128, heads 32/8 (G=4 even):

| case | gqa | short | varlen | splitk |
|---|---:|---:|---:|---:|
| prompt 512 | 588 | 2947 | 4026 | 6274 |
| prompt 1k | 2210 | 10698 | 14921 | 23906 |
| prompt 2k | 8562 | 40708 | 58369 | 93590 |
| prompt 4k | 34327 | 157409 | 227151 | 367992 |
| prompt 8k | 135684 | 627801 | 910217 | 1467741 |
| 4×prompt 1k | 8575 | 41035 | 58986 | 95712 |
| chunk 1k @ 8k | 32155 | 149169 | 212540 | 341565 |
| chunk 2k @ 16k | 125991 | 583030 | 842857 | 1358999 |

`gqa` wins **4–10×** at every shape — including the case the PR flagged as the
likely loss ("short chunks behind long prefixes", where splitk has more CTAs):
chunk 2k@16k is **10.8× faster** than splitk. G=7 (28/4, exercises `<128,1,16>`)
and D=256 G=12 show the same. `max|diff|` vs gqa ≤ **2.0e-3** (fp16 rounding).

⇒ The PR's open performance question is resolved in its favor on gfx1030: no
shape measured prefers the old kernels. Default-on routing is justified.

## Merge rationale

- Deployed models are D=256 even-group ⇒ bitwise-identical ⇒ zero regression risk.
- D=128 gains 4–10× kernel-level prefill, correctness within fp16 rounding, all
  shapes, including odd groups/MHA.
- 72/72 tests; clean merge with the open #33 (merge-tree, no conflicts).

## Artifacts (`remote/`)

`matrix_combined.csv` (all metrics), `bench_d128.log`, `bench_d256.log`,
`bench_clean_gpu5.out` (clean D=128/D=256 kernel A/B), `pytest_patched.log`,
`fa_meta.txt` (kernel descriptors), `k256_{baseline,patched}.s`
(the 2-instruction D=256 diff), per-arm `driver.log` / `coherence.txt` /
`cells/*.json`, `build_{baseline,patched}.log`, `bitwise_driver.sh`,
`probe_bitwise_gqa256.py`, `isa_compare.sh`, `pr34_ab.sh`, `run_matrix.sh`.
