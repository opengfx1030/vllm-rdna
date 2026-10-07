# Flash-Next MoE W4A8 A/B (gfx1030, 4x V620, TP=4)

MTP=0, FULL_AND_PIECEWISE, prefix caching, FA-RDNA2, RDNA_AR one-shot 64 KiB,
`VLLM_RDNA2_W4A8_SDOT4` 0 vs 1. Same tree, same launcher
(`scripts/serve_gfx1030_flashnext_mtp.sh`), fresh arm-tagged caches. Sequential
one-engine-at-a-time; PCI-SERR flat at 15 across both arms.

Model: `wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16`.

## Fast-path marker

`TORCH_WARN_ONCE("RDNA2 W4A8 sdot4 MoE path active (config moe_a8_k32_ag)")`:

* W4A8=1 arm: **4** occurrences (one per TP worker).
* W4A8=0 arm: **0**.

## Coherence

`tools/rdna2_028/probe_w4a8.py` (temperature 0): **4 OK / 0 BAD / 4 total** on
both arms.

## Cells

| cell | W4A8=1 out tok/s | W4A8=0 out tok/s | Δ | W4A8=1 TTFT ms | W4A8=0 TTFT ms | Δ TTFT | W4A8=1 prefill tok/s | W4A8=0 prefill tok/s | Δ prefill |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1k/512 c=1 | 41.01 | 41.78 | -1.8% | 572.5 | 619.2 | -7.5% | 1788.5 | 1653.7 | +8.2% |
| 1k/512 c=8 | 165.52 | 163.54 | +1.2% | 3691 | 3916 | -5.7% | 2219.3 | 2091.9 | +6.1% |
| 16k/1k c=1 | 31.64 | 31.29 | +1.1% | 8248 | 9140 | -9.8% | 1986.5 | 1792.5 | +10.8% |
| 16k/1k c=8 | 69.32 | 65.57 | +5.7% | 44346 | 48291 | -8.2% | 2955.7 | 2714.2 | +8.9% |

Per-cell raw logs: `on/cells.csv`, `off/cells.csv` (full bench `--save-result`
JSON in `on/cells/*/`, `off/cells/*/`).

## Verdict

**Real but modest — shipped opt-in, default stays W4A16.** The MoE GEMM itself
is 1.6-1.8x faster (see `probe.log`), which shows up as **+6 to +11% prefill
throughput and -6 to -10% TTFT** on every cell. Aggregate output throughput is
+1.1 to +5.7%: at low concurrency the tail is decode-dominated, where the MoE
GEMM is not the bottleneck, so the win dilutes. The `1k/512 c=1` -1.8% is within
run-to-run noise (the L2 micro measures M=64 at 1.01x). No corruption, no PCI
SERR, coherent on both arms.
