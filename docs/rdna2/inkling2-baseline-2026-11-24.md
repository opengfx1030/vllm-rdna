# Inkling-2-Mini serving baseline on `rdna_extras` 36a09895bb (2026-11-24)

Dated record. Full TP=1, TP=2, and TP=4 serving baseline (DSpark off and on) on `par1-cs25`, four-card slot `b4` (Radeon PRO V620, gfx1030, HIP 2–5, PCI `d9 dc df e2`), tree `rdna_extras` `36a09895bb` (runtime and `_rocm_C` identical to `d0b0c2b9af`), venv `venv-7.14.0_0.34.0-rebase-1e1cac5340`, native checkpoint `~/hfcache/inkling2-mini`. Driven by `~/w4a8_runs/inkling2-work/InklingBase/driver.sh` (`serve_validate.sh`, 2026-11-23 21:06 to 2026-11-24 00:43 CET).

## Boot and probe summary

| Run | Recipe / overrides | GPUs | Ready | `max_model_len` | KV cache tokens | Probes (greedy / PPL / prefix / concurrency / post-load) | KFD `evicted_ms` since READY | Max junction °C |
| --- | --- | --- | ---: | ---: | ---: | --- | --- | ---: |
| `tp1-off` | `inkling2-mini-tp1` | HIP 2 | 70 s | 180,000 | 180,756 | 3/3 / 5.21 / 15.74x / 0 fail (10 near-tie) / PASS | n/a (single-proc) | 78.0 |
| `tp1-on` | `inkling2-mini-tp1-dspark` (`SEQS=2`) | HIP 2 | 80 s | 125,000 | 125,597 | 3/3 / 5.21 / 18.68x / 0 fail (9 near-tie) / PASS | n/a (single-proc) | 79.0 |
| `tp2-off` | `inkling2-mini-tp2` | HIP 2,3 | 80 s | 1,048,576 | 1,062,482 | 3/3 / 5.20 / 25.04x / 0 fail (5 near-tie) / PASS | TP0 +0, TP1 +0 | 80.0 |
| `tp2-on` | `inkling2-mini-tp2-dspark` (`SEQS=2`) | HIP 2,3 | 100 s | 780,000 | 790,241 | 3/3 / 5.20 / 21.14x / 0 fail (12 near-tie) / PASS | TP0 +0, TP1 +0 | 81.0 |
| `tp4-off` | `inkling2-mini-tp4` | HIP 2,3,4,5 | 90 s | 1,048,576 | 2,247,316 | 3/3 / 5.19 / 15.82x / 0 fail (12 near-tie) / PASS | TP0..3 +0 | 83.0 |
| `tp4-on` | `inkling2-mini-tp4` + `SEQS=2 MAXLEN=780000` + `SPEC_CONFIG` (`[[1,2,2]]`) | HIP 2,3,4,5 | 110 s | 780,000 | 1,135,043 | 3/3 / 5.21 / 13.81x / 0 fail (9 near-tie) / PASS | TP0..3 +0 | 82.0 |

Zero fault markers (`HSA_STATUS_ERROR`, Memory Fault, Queue error, NaNs, `rdna_ar` wedge) in any `serve.log`. `tp4-on` is the first full serving validation of DSpark at TP=4 (using the TP=2 DSpark schedule; no dedicated TP=4 DSpark recipe exists yet).

## Baseline table

Random dataset, temperature 0, `ignore-eos`, prefix cache reset before every cell (67 cells total; 65 valid, 2 invalid after one rerun). `prefill tok/s` = mean prompt tokens / median TTFT (`c=1` is the per-request prefill rate; `c>1` includes queueing behind other prompts). `out tok/s` is aggregate output throughput over the cell.

| config | prompt/out | c | n | TTFT p50 ms | TTFT p99 ms | prefill tok/s (in/TTFT p50) | ITL p50 ms | ITL p99 ms | out tok/s (agg) | spec acc len | validity | build running / loadavg (start->end) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| TP1 off | 1024/512 | 1 | 6 | 493.4 | 501.0 | 2075.2 | 18.0 | 18.3 | 52.9 | - | ok | yes->yes / 16.2->14.4 |
| TP1 off | 1024/512 | 2 | 8 | 548.7 | 1000.4 | 1866.3 | 21.8 | 22.6 | 84.4 | - | ok [rerun] | yes->yes / 11.8->12.1 |
| TP1 off | 1024/512 | 4 | 16 | 1508.0 | 1959.0 | 679.0 | 27.6 | 29.0 | 127.6 | - | ok [rerun] | yes->yes / 12.1->12.1 |
| TP1 off | 1024/512 | 8 | 32 | 1979.9 | 3934.8 | 517.2 | 37.2 | 63.1 | 178.3 | - | ok [rerun] | yes->yes / 12.1->12.3 |
| TP1 off | 16384/512 | 1 | 4 | 8518.8 | 8660.4 | 1923.3 | 19.2 | 19.8 | 27.9 | - | ok | yes->yes / 12.8->16.1 |
| TP1 off | 16384/512 | 2 | 4 | 10103.3 | 16780.0 | 1621.6 | 24.2 | 533.8 | 35.0 | - | ok | yes->yes / 13.6->15.9 |
| TP1 off | 16384/512 | 4 | 8 | 10569.1 | 34250.0 | 1550.2 | 31.9 | 581.1 | 40.5 | - | ok | yes->yes / 15.9->14.1 |
| TP1 off | 16384/512 | 8 | 8 | 39799.2 | 70285.4 | 411.7 | 45.0 | 590.6 | 44.3 | - | ok | yes->yes / 14.1->14.3 |
| TP1 off | 65536/256 | 1 | 3 | 43314.5 | 43508.6 | 1513.0 | 22.9 | 23.2 | 5.2 | - | ok | yes->yes / 13.1->14.5 |
| TP1 off | 65536/256 | 2 | 2 | 65355.2 | 86990.9 | 1002.8 | 31.6 | 839.8 | 5.4 | - | ok | yes->yes / 14.4->12.9 |
| TP1 DSpark | 1024/512 | 1 | 6 | 512.9 | 529.2 | 1996.4 | 30.3 | 32.0 | 77.8 | 2.55 | ok | yes->yes / 12.2->12.2 |
| TP1 DSpark | 1024/512 | 2 | 8 | 603.1 | 980.4 | 1697.8 | 38.7 | 42.3 | 112.4 | 2.49 | ok | yes->yes / 12.2->12.1 |
| TP1 DSpark | 1024/512 | 4 | 16 | 9173.3 | 14467.1 | 111.6 | 38.4 | 41.2 | 110.2 | 2.38 | ok | yes->yes / 12.1->12.4 |
| TP1 DSpark | 1024/512 | 8 | 32 | 26247.8 | 33734.1 | 39.0 | 38.4 | 43.1 | 112.3 | 2.44 | ok | yes->yes / 12.4->12.6 |
| TP1 DSpark | 16384/512 | 1 | 4 | 8650.4 | 8748.7 | 1894.0 | 33.6 | 34.6 | 32.5 | 2.39 | ok | yes->yes / 10.7->11.8 |
| TP1 DSpark | 16384/512 | 2 | 4 | 9180.8 | 17358.9 | 1784.6 | 42.9 | 586.9 | 37.9 | 2.38 | ok | yes->yes / 11.8->12.1 |
| TP1 DSpark | 16384/512 | 4 | 8 | 34907.6 | 44960.5 | 469.4 | 42.8 | 592.4 | 35.9 | 2.16 | ok | yes->yes / 12.1->12.7 |
| TP1 DSpark | 16384/512 | 8 | 8 | 53345.3 | 96174.2 | 307.1 | 42.9 | 594.2 | 36.2 | 2.16 | ok | yes->yes / 12.7->12.7 |
| TP1 DSpark | 65536/256 | 1 | 3 | 43498.1 | 43663.2 | 1506.6 | 42.3 | 43.8 | 5.3 | 2.22 | ok | yes->yes / 11.0->12.3 |
| TP2 off | 1024/512 | 1 | 6 | 603.0 | 614.0 | 1698.2 | 12.8 | 13.2 | 71.7 | - | ok | yes->yes / 11.0->10.2 |
| TP2 off | 1024/512 | 2 | 8 | 730.4 | 1183.1 | 1401.9 | 14.9 | 16.1 | 116.6 | - | ok | yes->yes / 10.2->9.7 |
| TP2 off | 1024/512 | 4 | 16 | 833.3 | 2316.3 | 1228.8 | 18.0 | 109.0 | 178.4 | - | ok | yes->yes / 9.7->9.9 |
| TP2 off | 1024/512 | 8 | 32 | 923.2 | 4773.2 | 1109.2 | 24.3 | 114.7 | 244.4 | - | ok | yes->yes / 9.9->9.2 |
| TP2 off | 16384/512 | 1 | 4 | 10020.4 | 10101.8 | 1635.1 | 13.4 | 13.9 | 30.4 | - | ok | yes->yes / 9.2->8.7 |
| TP2 off | 16384/512 | 2 | 4 | 10680.1 | 19574.9 | 1534.1 | 16.0 | 119.3 | 37.2 | - | ok | yes->yes / 4.2->7.6 |
| TP2 off | 16384/512 | 4 | 8 | 11969.5 | 41252.6 | 1368.8 | 20.6 | 134.8 | 40.5 | - | ok | yes->yes / 7.6->9.1 |
| TP2 off | 16384/512 | 8 | 8 | 47968.3 | 88037.2 | 341.6 | 118.8 | 137.9 | 41.6 | - | ok | yes->yes / 9.1->9.2 |
| TP2 off | 65536/256 | 1 | 3 | 44673.0 | 44859.3 | 1467.0 | 15.4 | 15.9 | 5.3 | - | ok | yes->yes / 6.9->10.7 |
| TP2 off | 65536/256 | 2 | 2 | 68333.7 | 91415.0 | 959.1 | 97.7 | 145.3 | 5.3 | - | ok | yes->yes / 10.7->9.5 |
| TP2 off | 65536/256 | 4 | 4 | 115520.4 | 185118.3 | 567.3 | 123.9 | 146.8 | 5.4 | - | ok | yes->yes / 9.2->9.4 |
| TP2 off | 65536/256 | 8 | 8 | 208933.9 | 370954.5 | 313.7 | 126.8 | 146.2 | 5.4 | - | ok | yes->yes / 3.4->8.4 |
| TP2 DSpark | 1024/512 | 1 | 6 | 623.7 | 637.8 | 1641.8 | 21.5 | 22.2 | 106.3 | 2.63 | ok | yes->yes / 7.0->6.8 |
| TP2 DSpark | 1024/512 | 2 | 8 | 705.6 | 1241.6 | 1451.3 | 26.1 | 112.7 | 110.5 | 1.92 | ok | yes->yes / 7.0->7.5 |
| TP2 DSpark | 1024/512 | 4 | 16 | 7094.8 | 13099.2 | 144.3 | 26.3 | 114.6 | 143.7 | 2.28 | ok | yes->yes / 7.5->7.4 |
| TP2 DSpark | 1024/512 | 8 | 32 | 20809.9 | 26740.5 | 49.2 | 26.3 | 115.0 | 145.9 | 2.32 | ok | yes->yes / 7.4->8.3 |
| TP2 DSpark | 16384/512 | 1 | 4 | 9943.5 | 10081.1 | 1647.7 | 23.9 | 24.4 | 34.1 | 2.39 | ok | yes->yes / 3.7->6.6 |
| TP2 DSpark | 16384/512 | 2 | 4 | 11486.9 | 20888.0 | 1426.3 | 29.1 | 128.4 | 38.1 | 2.42 | ok | yes->yes / 6.6->7.5 |
| TP2 DSpark | 16384/512 | 4 | 8 | 36746.8 | 47666.1 | 445.9 | 29.3 | 129.0 | 36.4 | 2.10 | ok | yes->yes / 7.5->8.2 |
| TP2 DSpark | 16384/512 | 8 | 8 | 55196.7 | 99569.1 | 296.8 | 29.8 | 129.3 | 36.7 | 2.19 | ok | yes->yes / 8.2->8.2 |
| TP2 DSpark | 65536/256 | 1 | 3 | 44921.1 | 45118.0 | 1458.9 | 30.3 | 31.2 | 5.3 | 2.30 | ok | yes->yes / 3.3->7.9 |
| TP2 DSpark | 65536/256 | 2 | 2 | 69231.3 | 92970.6 | 946.6 | 30.6 | 152.7 | 5.3 | 2.33 | ok | yes->yes / 7.9->7.8 |
| TP2 DSpark | 65536/256 | 4 | 4 | 119336.9 | 191815.3 | 549.2 | 150.1 | 155.3 | 5.2 | 2.26 | ok | yes->yes / 7.8->8.4 |
| TP2 DSpark | 65536/256 | 8 | 8 | 215785.8 | 385384.5 | 303.7 | 150.8 | 155.2 | 5.2 | 2.48 | ok | no->no / 4.3->8.5 |
| TP4 off | 1024/512 | 1 | 6 | 329.0 | 330.2 | 3112.5 | 10.0 | 10.2 | 94.2 | - | ok | no->no / 9.0->9.8 |
| TP4 off | 1024/512 | 2 | 8 | 496.2 | 667.4 | 2063.8 | 11.4 | 11.8 | 158.8 | - | ok | no->no / 9.2->9.2 |
| TP4 off | 1024/512 | 4 | 16 | 1274.7 | 1287.8 | 803.3 | 13.4 | 13.9 | 252.5 | - | ok | no->no / 9.2->10.1 |
| TP4 off | 1024/512 | 8 | 32 | 1599.1 | 2544.6 | 640.3 | 17.5 | 18.2 | 355.8 | - | ok | no->no / 10.1->10.1 |
| TP4 off | 16384/512 | 1 | 4 | 5291.7 | 5327.5 | 3096.2 | 10.3 | 10.6 | 48.5 | - | ok | no->no / 10.1->10.5 |
| TP4 off | 16384/512 | 2 | 4 | 7909.6 | 10523.6 | 2071.4 | 12.0 | 12.4 | 61.6 | - | ok | no->no / 10.5->9.2 |
| TP4 off | 16384/512 | 4 | 8 | 11688.0 | 20516.0 | 1401.8 | 14.7 | 1282.4 | 72.3 | - | ok | no->no / 3.7->12.6 |
| TP4 off | 16384/512 | 8 | 8 | 24676.7 | 41534.4 | 663.9 | 19.7 | 1322.1 | 79.2 | - | ok [rerun] | no->no / 10.3->11.8 |
| TP4 off | 65536/256 | 1 | 3 | 23446.6 | 23632.8 | 2795.1 | 11.4 | 12.0 | 9.7 | - | ok [rerun] | no->no / 11.8->13.7 |
| TP4 off | 65536/256 | 2 | 2 | 35100.8 | 46560.8 | 1867.1 | 14.0 | 1521.7 | 10.2 | - | INVALID: boot.lock held during the cell [rerun] | no->no / 13.7->8.9 |
| TP4 off | 65536/256 | 4 | 4 | 59843.7 | 93259.8 | 1095.1 | 18.3 | 1616.2 | 10.4 | - | INVALID: boot.lock held during the cell [rerun] | no->no / 8.9->8.2 |
| TP4 off | 65536/256 | 8 | 8 | 106112.6 | 186132.0 | 617.6 | 26.9 | 1647.2 | 10.6 | - | ok [rerun] | no->no / 4.2->11.1 |
| TP4 DSpark | 1024/512 | 1 | 6 | 340.8 | 342.5 | 3005.1 | 17.5 | 17.9 | 145.5 | 2.80 | ok | no->no / 7.2->6.9 |
| TP4 DSpark | 1024/512 | 2 | 8 | 370.3 | 660.9 | 2765.6 | 20.4 | 21.1 | 154.7 | 2.05 | ok | no->no / 6.9->7.2 |
| TP4 DSpark | 1024/512 | 4 | 16 | 4982.7 | 7807.5 | 205.5 | 20.6 | 21.4 | 194.8 | 2.33 | ok | no->no / 6.7->8.4 |
| TP4 DSpark | 1024/512 | 8 | 32 | 14571.7 | 18842.4 | 70.3 | 20.6 | 21.9 | 202.2 | 2.40 | ok | no->no / 8.4->9.7 |
| TP4 DSpark | 16384/512 | 1 | 4 | 5333.8 | 5365.3 | 3071.8 | 19.8 | 20.2 | 53.7 | 2.40 | ok | no->no / 9.7->9.2 |
| TP4 DSpark | 16384/512 | 2 | 4 | 7861.1 | 10362.2 | 2084.2 | 23.1 | 53.6 | 66.6 | 2.43 | ok | no->no / 3.6->5.5 |
| TP4 DSpark | 16384/512 | 4 | 8 | 20040.5 | 25888.1 | 817.5 | 23.1 | 1290.2 | 61.8 | 2.16 | ok | no->no / 1.7->6.9 |
| TP4 DSpark | 16384/512 | 8 | 8 | 31044.9 | 55910.0 | 527.8 | 23.0 | 1306.0 | 61.6 | 2.13 | ok | no->no / 7.5->8.6 |
| TP4 DSpark | 65536/256 | 1 | 3 | 23516.2 | 23607.3 | 2786.8 | 25.5 | 26.0 | 9.7 | 2.32 | ok | no->no / 8.0->8.4 |
| TP4 DSpark | 65536/256 | 2 | 2 | 35309.2 | 46942.3 | 1856.1 | 29.8 | 1603.2 | 10.1 | 2.35 | ok | no->no / 8.2->7.9 |
| TP4 DSpark | 65536/256 | 4 | 4 | 59919.5 | 95146.3 | 1093.7 | 30.0 | 1602.0 | 10.3 | 2.50 | ok | no->no / 3.2->7.5 |
| TP4 DSpark | 65536/256 | 8 | 8 | 109682.5 | 195459.2 | 597.5 | 30.0 | 1626.1 | 10.3 | 2.48 | ok | no->no / 1.7->11.3 |

## Observations and caveats

- **Random dataset and DSpark acceptance**: `serve_validate.sh` sends random-token prompts. DSpark acceptance length (2.05–2.80 here) reflects random-prompt repetition and is not representative of real text; real-text acceptance and sampling checks are run separately (see [`inkling2-speculation-profile.md`](inkling2-speculation-profile.md)).
- **DSpark concurrency cliff (`SEQS=2`)**: all three DSpark runs use `SEQS=2` (because draft CUDA graphs currently capture up to `max_num_seqs` regardless of the batch-size schedule). At `1024/512 c=8`, aggregate output throughput drops -37% at TP=1 (`178.3 -> 112.3 tok/s`, TTFT `2.0 -> 26.2 s`), -40% at TP=2 (`244.4 -> 145.9 tok/s`, TTFT `0.9 -> 20.8 s`), and -43% at TP=4 (`355.8 -> 202.2 tok/s`, TTFT `1.6 -> 14.6 s`).
- **TP=2 and TP=4 prefill gap**: `TP2 off` prefill at `1024/512 c=1` (`1698.2 tok/s`) is slower than `TP1 off` (`2075.2 tok/s`), because the 32-row NVFP4 prefill kernel (`moe_prefill_bm_rdna2.cu`) requires `K % 128 == 0` and falls back to the decode kernel at TP=2 (`I=576`) and TP=4 (`I=288`), compounded at TP=2 by `MAXBAT=184`.
- **DSpark 64k ITL jump**: at `65536/256 c=1`, ITL increases from `30.3 -> 42.3 ms` (+12.0 ms) at TP=1 DSpark vs `18.0 -> 22.9 ms` (+4.9 ms) with DSpark off (the draft's full-attention layer 1 attends over all 64k context keys).
- **Window caveats**:
    - A Docker build ran concurrently on the host during `tp1-off`, `tp1-on`, `tp2-off`, and most of `tp2-on` (recorded in the rightmost column; `loadavg1` 3.3–16.6). `tp4-off` and `tp4-on` ran after the build finished (`build=no`).
    - `tp1-off` had 3 cells invalidated by `boot.lock` held on `a2` and rerun cleanly (`tp1-off-r2`).
    - `tp4-off` had 5 cells invalidated by `boot.lock`; 3 cleared on rerun (`tp4-off-r2`) while `65536/256 c=2` and `c=4` remained `INVALID` because `boot.lock` was held again by `a2` during the rerun.
    - At TP=1 the KFD worker scraper prints `no vLLM workers with a KFD context found` (single-process `EngineCore` naming); TP=2 and TP=4 workers all reported `+0 ms` KFD evictions since READY.
    - The kernel log is unreadable on `par1-cs25`, so `dmesg` faults are unknown, not zero.
