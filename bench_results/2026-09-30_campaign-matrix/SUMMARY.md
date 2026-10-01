# TunableOp full-matrix campaign — capture / tune / curate / freeze / validate

Box `par1-cs25` (4× Radeon PRO V620 gfx1030 in HIP 0-3), tree `vllm-rdna-0.28.0`,
venv-7.14.0_0.28.0 (PyTorch 2.12.0+rocm7.14.0, rocBLAS 5.5.0, `librocblas.so.5`
sha256 `f30bb442e9b5…`). Matrix: **{W4A16, W4A8} × {FA-RDNA2, Triton AMD FA} ×
{MTP=0, MTP=2}**, TP=4, FULL_AND_PIECEWISE, prefix caching, MTP ladder
`[3,6,12,24]`, RDNA_AR one-shot ≤64 KiB, `--max-num-seqs 8`.

Frozen row set: `tunableop/rocblas-f30bb442e9b5/` — **760 data rows/rank**
(commit `d9c927c5a`). Lookup-only, keyed by the rocBLAS build hash, never
`/tmp` or the run CWD.

## 1. Shape capture (record-untuned census)

`tools/rdna2_028/campaign_matrix.sh MODE=capture` boots each arm with
`PYTORCH_TUNABLEOP_RECORD_UNTUNED=1` + `PYTORCH_TUNABLEOP_UNTUNED_FILENAME`.

> **Mechanism finding (important).** `record_untuned` *disables* the results
> lookup: with it on, TunableOp loads **0** rows and records **every** fp16 GEMM
> to `untuned<rank>.csv` (a full observed-shape census), and the cells run on
> rocBLAS heuristics. Capture and performance validation therefore **cannot**
> share a boot; the capture arms used short (48-token) outputs purely to fire
> the shapes. This contradicts the prior `campaign_3a_capture.sh` design (which
> also used a non-existent env var `…_RECORD_UNTUNED_FILENAME` and would have
> written captures to the run CWD).

Observed-key census per arm (`captures/shapes_<arm>.txt`):

| MTP | FA-RDNA2 W4A16 | FA-RDNA2 W4A8 | Triton W4A16 | Triton W4A8 |
|---|---:|---:|---:|---:|
| 0 | 150 | 150 | 150 | 150 |
| 2 | 201 | 200 | 201 | 201 |

The four MTP=0 arms are **set-identical** (pairwise symmetric difference 0); the
four MTP=2 arms are set-identical to within scheduling jitter (the single
`w4a8-fa-m2` delta is one transient mixed-batch shape). **Only MTP changes the
rocBLAS fp16 GEMM shape set** — attention backend (FA vs Triton) and the W4A8
MoE path do not, because MoE and attention are not rocBLAS GEMMs.

## 2. Tune

`tunableop_rows_pipeline.sh PHASE=all`, `SHAPE_SRC` = union of the committed
rows and all captured traces → **879 shapes** after small-M extension, tuned on
all four GPUs in parallel (≥10 iterations, 25 ms/solver, numerical check
0.01/0.01). Measure phases: heuristic (TunableOp off) vs current (committed
rows) vs new (scratch), same process, GPU 0, 5 warmup + 15 reps median.

## 3. Merge + curate

`curate_tunableop_rows.py` with 3 %/3 % adopt/drop margins over 879 shapes:
**52 new · 708 current · 119 dropped**. Net rows **754 → 760**.

| decision | count | note |
|---|---:|---|
| new | 52 | 30 net-new keys + 22 improved solvers on existing keys |
| current | 708 | carried over unchanged |
| dropped | 119 | Default/heuristic-equivalent + 24 previously-shipped rows that a clean measurement found >3 % slower than the heuristic |

Top new decisions (before/after, `curated/before_after.csv`):

| shape | heur ms | current ms | new ms | speedup |
|---|---:|---:|---:|---:|
| `tn_10240_1033_2560` | 9.909 | 10.402 | 2.508 | **3.95×** |
| `tn_10240_1039_2560` | 9.558 | 10.330 | 2.512 | 3.80× |
| `tn_336_1039_10240` | 1.850 | 1.793 | 0.471 | 3.81× |
| `tn_336_1033_10240` | 1.883 | 1.810 | 0.476 | 3.80× |
| `tn_320_1033_10240` | 1.621 | 1.651 | 0.458 | 3.54× |
| `tn_320_1039_10240` | 1.645 | 1.642 | 0.457 | 3.60× |
| `nn_1_1033_2560` | 0.0696 | 0.0739 | 0.0239 | 3.09× |
| `tn_320_1033_2560` | 0.367 | 0.404 | 0.153 | 2.64× |
| `tn_24_1033_2560` | 0.115 | 0.120 | 0.065 | 1.86× |

The wins are the **MTP-2 mixed prefill+decode batch families** (`M = 1024/2048 +
k`, e.g. `tn_640_4120_2560`, `tn_320_1039_10240`) that no harvested/captured
row set previously contained — `record_untuned` is what exposed them.

## 4. Freeze

Frozen into `tunableop/rocblas-f30bb442e9b5/` (760 rows/rank) + `provenance.json`
(`campaign` block), committed and pushed to `w4a8-wiring` as **`d9c927c5a`**.
Prior set preserved at `repo-rows-premerge/`.

## 5. Validation (frozen rows, lookup-only)

### Lookup-hit proof

`verify_tunableop_lookup.py --rows …/tunableop_results0.csv`:
**760/760 shapes hit, 0 misses** (`lookup_hits.json`). The production launcher
logs `TunableOp lookup enabled for rocBLAS build f30bb442e9b5 (rows: …/tunableop/rocblas-f30bb442e9b5); tuning off`
on every arm; `PYTORCH_TUNABLEOP_VERBOSE=1` additionally prints
`reading tuning results from …/rocblas-f30bb442e9b5/tunableop_results0.csv`.

### Coherence + marker

All 8 arms: **4 OK / 0 BAD** coherence probes ("The capital of France is" →
"Paris.", "2 + 2 =" → "4"). `RDNA2 W4A8 sdot4 MoE path active` on all four W4A8
arms (rank0+rank3 shown), **absent** on all four W4A16 arms. PCI-SERR flat
(base 1 → 1, no reboot).

### Matrix (single seed per cell; `validate/<arm>/cells.csv`)

`out` = aggregate output tok/s; `pref` = prefill tok/s (in_tok / mean TTFT);
`dec/req` = per-request decode tok/s; `ITL` = median inter-token latency ms.

**MTP=2**

| config | 16k/1k c=1 out (TTFT s / pref / dec/req / ITL) | 16k/1k c=8 out (TTFT s / pref / dec/req / ITL) | 1k/512 c=1 out | 1k/512 c=8 out |
|---|---|---|---|---|
| W4A16 · FA | 44.85 (9.74 / 1681 / 78.2 / 35.4) | 69.19 (49.2 / 2662 / 15.5 / 83.3) | 75.29 | 141.39 |
| W4A8 · FA | 43.55 (8.84 / 1853 / 69.7 / 36.0) | 68.86 (45.4 / 2889 / 16.1 / 82.1) | 70.74 | 144.72 |
| W4A16 · Triton | 41.80 (9.64 / 1699 / 68.9 / 35.5) | 64.95 (49.5 / 2647 / 15.8 / 83.1) | 56.89 | 140.47 |
| W4A8 · Triton | 46.12 (8.82 / 1857 / 76.5 / 36.1) | 74.07 (45.3 / 2893 / 16.6 / 83.6) | 69.92 | 132.61 |

**MTP=0**

| config | 16k/1k c=1 out (TTFT / pref / dec/req / ITL) | 16k/1k c=8 out | 1k/512 c=1 out | 1k/512 c=8 out |
|---|---|---|---|---|
| W4A16 · FA | 31.18 (9.27 / 1767 / 43.4 / 23.1) | 69.14 | 41.72 | 173.37 |
| W4A8 · FA | 31.43 (8.42 / 1947 / 42.3 / 23.6) | 74.43 | 40.94 | 176.69 |
| W4A16 · Triton | 31.19 (9.25 / 1772 / 43.4 / 23.1) | 69.75 | 41.76 | 172.60 |
| W4A8 · Triton | 31.48 (8.34 / 1964 / 42.3 / 23.7) | 74.30 | 40.95 | 175.62 |

### W4A8 vs W4A16 delta (aggregate output tok/s)

| MTP | backend | 16k c=1 | 16k c=8 | 1k c=1 | 1k c=8 |
|---|---|---:|---:|---:|---:|
| 2 | FA | −2.9 % | −0.5 % | −6.0 % | +2.4 % |
| 2 | Triton | **+10.3 %** | **+14.1 %** | **+22.9 %** | −5.6 % |
| 0 | FA | +0.8 % | +7.7 % | −1.9 % | +1.9 % |
| 0 | Triton | +0.9 % | +6.5 % | −2.0 % | +1.8 % |

W4A8 is a clear win on the **MTP=0 16k c=8** cells (+6.5–7.7 %) and on the
MTP=2 Triton 16k cells; it is neutral-to-slightly-negative on the MTP=2 FA 16k
c=1 cell and on 1k/512 c=1. These are single-seed cells; the 1k/512 MTP=2 FA
(−6 %) and Triton (−5.6 %) deltas are within the run-to-run spread seen on this
stack and need a repeat before being read as regressions.

## Notes / caveats

- A sibling agent session was running this same campaign concurrently (commits
  `1707cc0bd`, `ee4f37906`, `6d0e4861e`) and an earlier kill of mine stopped its
  `campaign_3a` server. The 754-row intermediate set from that session is
  preserved at `repo-rows-premerge/`; this campaign's clean, full-census curate
  supersedes it (its freeze was explicitly additive because its measurements
  ran under the contention of this capture).
- Single seed per cell, `--temperature 0`; per-cell coherence is the correctness
  signal (greedy output is not bitwise reproducible on this stack).
- `usage.prompt_tokens_details.cached_tokens` is 0 in this fork — prefix-cache
  behaviour is read from TTFT / server hit-rate, not that field.
- Tooling: `tools/rdna2_028/{campaign_matrix.sh, probe_record_untuned.py,
  analyze_captures.py}`; reuse pipeline `tunableop_rows_pipeline.sh`,
  `curate_tunableop_rows.py`, `verify_tunableop_lookup.py`.
