# HWQ ABBA: GPU_MAX_HW_QUEUES 2 (A) vs 4 (B), flashnext-mtp2 TP4, slot B

Arms in run order: A1 (2), B1 (4), B2 (4), A2 (2). Fresh boot per arm, same tree, greedy, same seeds.

## Per-arm cells

### A1 (GPU_MAX_HW_QUEUES=2)

| cell | TTFT med ms | TTFT p99 ms | ITL med ms | ITL mean ms | ITL p99 ms | out tok/s | MTP acc len | prefill agg tok/s | KFD / reruns |
|---|---|---|---|---|---|---|---|---|---|
| 1024x512-c1 | 380 | 470 | 23.75 | 23.74 | 24.2 | 66.4 | 1.50 | 2236 | evicted_ms Worker_TP0_EP0=+0 Worker_TP1_EP1=+0 Worker_TP2_EP; invalid runs 0, seed 12301 |
| 1024x512-c6 | 522 | 2228 | 42.87 | 46.75 | 372.0 | 204.2 | 1.90 | 2757 | evicted_ms Worker_TP0_EP0=+0 Worker_TP1_EP1=+0 Worker_TP2_EP; invalid runs 0, seed 12302 |
| 1024x512-c8 | 540 | 3006 | 45.67 | 52.70 | 380.1 | 272.0 | 2.06 | 2725 | evicted_ms Worker_TP0_EP0=+0 Worker_TP1_EP1=+0 Worker_TP2_EP; invalid runs 0, seed 12303 |
| 16384x1024-c1 | 5656 | 5685 | 24.43 | 24.38 | 25.0 | 64.7 | 2.46 | 2882 | evicted_ms Worker_TP0_EP0=+0 Worker_TP1_EP1=+0 Worker_TP2_EP; invalid runs 0, seed 12304 |
| 16384x1024-c8 | 15726 | 50261 | 48.07 | 119.35 | 427.1 | 114.4 | 2.39 | 2701 | evicted_ms Worker_TP0_EP0=+0 Worker_TP1_EP1=+0 Worker_TP2_EP; invalid runs 0, seed 12305 |

### B1 (GPU_MAX_HW_QUEUES=4)

| cell | TTFT med ms | TTFT p99 ms | ITL med ms | ITL mean ms | ITL p99 ms | out tok/s | MTP acc len | prefill agg tok/s | KFD / reruns |
|---|---|---|---|---|---|---|---|---|---|
| 1024x512-c1 | 386 | 471 | 22.62 | 22.63 | 23.2 | 85.4 | 2.07 | 2230 | evicted_ms Worker_TP0_EP0=+0 Worker_TP1_EP1=+0 Worker_TP2_EP; invalid runs 0, seed 12301 |
| 1024x512-c6 | 518 | 2240 | 41.38 | 46.02 | 374.8 | 216.7 | 1.93 | 2742 | evicted_ms Worker_TP0_EP0=+0 Worker_TP1_EP1=+0 Worker_TP2_EP; invalid runs 0, seed 12302 |
| 1024x512-c8 | 538 | 2999 | 44.52 | 52.21 | 379.5 | 282.3 | 2.13 | 2731 | evicted_ms Worker_TP0_EP0=+0 Worker_TP1_EP1=+0 Worker_TP2_EP; invalid runs 0, seed 12303 |
| 16384x1024-c1 | 5643 | 5678 | 23.22 | 23.18 | 23.7 | 67.2 | 2.47 | 2886 | evicted_ms Worker_TP0_EP0=+0 Worker_TP1_EP1=+0 Worker_TP2_EP; invalid runs 0, seed 12304 |
| 16384x1024-c8 | 8631 | 49803 | 46.68 | 122.49 | 424.0 | 115.0 | 2.32 | 2726 | evicted_ms Worker_TP0_EP0=+0 Worker_TP1_EP1=+0 Worker_TP2_EP; invalid runs 0, seed 12305 |

### B2 (GPU_MAX_HW_QUEUES=4)

| cell | TTFT med ms | TTFT p99 ms | ITL med ms | ITL mean ms | ITL p99 ms | out tok/s | MTP acc len | prefill agg tok/s | KFD / reruns |
|---|---|---|---|---|---|---|---|---|---|
| 1024x512-c1 | 383 | 474 | 22.69 | 22.69 | 23.2 | 87.4 | 2.12 | 2215 | evicted_ms Worker_TP0_EP0=+0 Worker_TP1_EP1=+0 Worker_TP2_EP; invalid runs 0, seed 12301 |
| 1024x512-c6 | 521 | 2238 | 42.01 | 46.09 | 375.1 | 220.7 | 2.05 | 2746 | evicted_ms Worker_TP0_EP0=+0 Worker_TP1_EP1=+0 Worker_TP2_EP; invalid runs 0, seed 12302 |
| 1024x512-c8 | 535 | 3000 | 44.45 | 51.63 | 379.7 | 267.3 | 2.00 | 2731 | evicted_ms Worker_TP0_EP0=+0 Worker_TP1_EP1=+0 Worker_TP2_EP; invalid runs 0, seed 12303 |
| 16384x1024-c1 | 5645 | 5674 | 23.07 | 23.06 | 23.7 | 60.3 | 2.05 | 2887 | evicted_ms Worker_TP0_EP0=+0 Worker_TP1_EP1=+0 Worker_TP2_EP; invalid runs 0, seed 12304 |
| 16384x1024-c8 | 7446 | 49349 | 46.37 | 121.17 | 408.4 | 114.2 | 2.22 | 2751 | evicted_ms Worker_TP0_EP0=+0 Worker_TP1_EP1=+0 Worker_TP2_EP; invalid runs 0, seed 12305 |

### A2 (GPU_MAX_HW_QUEUES=2)

| cell | TTFT med ms | TTFT p99 ms | ITL med ms | ITL mean ms | ITL p99 ms | out tok/s | MTP acc len | prefill agg tok/s | KFD / reruns |
|---|---|---|---|---|---|---|---|---|---|
| 1024x512-c1 | 389 | 474 | 24.78 | 24.78 | 25.3 | 75.6 | 1.94 | 2218 | evicted_ms Worker_TP0_EP0=+0 Worker_TP1_EP1=+0 Worker_TP2_EP; invalid runs 0, seed 12301 |
| 1024x512-c6 | 531 | 2265 | 44.46 | 48.83 | 378.9 | 210.9 | 2.11 | 2712 | evicted_ms Worker_TP0_EP0=+0 Worker_TP1_EP1=+0 Worker_TP2_EP; invalid runs 0, seed 12302 |
| 1024x512-c8 | 548 | 3038 | 47.26 | 54.46 | 384.1 | 256.9 | 2.07 | 2696 | evicted_ms Worker_TP0_EP0=+0 Worker_TP1_EP1=+0 Worker_TP2_EP; invalid runs 0, seed 12303 |
| 16384x1024-c1 | 5690 | 5721 | 25.32 | 25.26 | 25.9 | 63.9 | 2.50 | 2864 | evicted_ms Worker_TP0_EP0=+0 Worker_TP1_EP1=+0 Worker_TP2_EP; invalid runs 0, seed 12304 |
| 16384x1024-c8 | 8476 | 50623 | 49.60 | 129.28 | 429.0 | 114.0 | 2.41 | 2682 | evicted_ms Worker_TP0_EP0=+0 Worker_TP1_EP1=+0 Worker_TP2_EP; invalid runs 0, seed 12305 |

## A vs B (mean of A1/A2 vs mean of B1/B2; spread = |A1-A2|, |B1-B2|)

| cell | metric | A1 | A2 | B1 | B2 | mean A | mean B | B-A | B vs A % | spread A | spread B | beyond spread |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1024x512-c1 | TTFT median | 380.32 | 389.17 | 385.59 | 382.98 | 384.74 | 384.29 | -0.45 | -0.1 | 8.85 | 2.61 | within spread |
| 1024x512-c1 | TTFT p99 | 469.85 | 473.56 | 471.10 | 474.18 | 471.70 | 472.64 | +0.94 | +0.2 | 3.71 | 3.08 | within spread |
| 1024x512-c1 | ITL median | 23.75 | 24.78 | 22.62 | 22.69 | 24.26 | 22.66 | -1.61 | -6.6 | 1.03 | 0.06 | better |
| 1024x512-c1 | ITL mean | 23.74 | 24.78 | 22.63 | 22.69 | 24.26 | 22.66 | -1.60 | -6.6 | 1.04 | 0.06 | better |
| 1024x512-c1 | ITL p99 | 24.21 | 25.29 | 23.21 | 23.22 | 24.75 | 23.22 | -1.53 | -6.2 | 1.08 | 0.02 | better |
| 1024x512-c1 | output tok/s | 66.37 | 75.63 | 85.42 | 87.43 | 71.00 | 86.43 | +15.43 | +21.7 | 9.26 | 2.02 | better |
| 1024x512-c1 | MTP acceptance length | 1.50 | 1.94 | 2.07 | 2.12 | 1.72 | 2.10 | +0.38 | +22.1 | 0.45 | 0.05 | within spread |
| 1024x512-c1 | prefill aggregate | 2235.84 | 2218.31 | 2229.90 | 2215.40 | 2227.07 | 2222.65 | -4.43 | -0.2 | 17.53 | 14.50 | within spread |
| 1024x512-c6 | TTFT median | 522.00 | 531.24 | 517.75 | 520.60 | 526.62 | 519.17 | -7.45 | -1.4 | 9.24 | 2.86 | within spread |
| 1024x512-c6 | TTFT p99 | 2228.44 | 2265.25 | 2240.33 | 2237.63 | 2246.85 | 2238.98 | -7.87 | -0.4 | 36.81 | 2.71 | within spread |
| 1024x512-c6 | ITL median | 42.87 | 44.46 | 41.38 | 42.01 | 43.67 | 41.69 | -1.97 | -4.5 | 1.59 | 0.63 | better |
| 1024x512-c6 | ITL mean | 46.75 | 48.83 | 46.02 | 46.09 | 47.79 | 46.05 | -1.74 | -3.6 | 2.09 | 0.08 | within spread |
| 1024x512-c6 | ITL p99 | 372.03 | 378.94 | 374.82 | 375.12 | 375.49 | 374.97 | -0.52 | -0.1 | 6.91 | 0.30 | within spread |
| 1024x512-c6 | output tok/s | 204.23 | 210.90 | 216.74 | 220.71 | 207.57 | 218.72 | +11.16 | +5.4 | 6.67 | 3.96 | better |
| 1024x512-c6 | MTP acceptance length | 1.90 | 2.11 | 1.93 | 2.05 | 2.00 | 1.99 | -0.01 | -0.6 | 0.22 | 0.11 | within spread |
| 1024x512-c6 | prefill aggregate | 2757.08 | 2712.28 | 2742.45 | 2745.77 | 2734.68 | 2744.11 | +9.43 | +0.3 | 44.80 | 3.32 | within spread |
| 1024x512-c8 | TTFT median | 540.28 | 548.06 | 538.32 | 534.53 | 544.17 | 536.43 | -7.74 | -1.4 | 7.77 | 3.79 | within spread |
| 1024x512-c8 | TTFT p99 | 3006.34 | 3038.26 | 2999.37 | 2999.53 | 3022.30 | 2999.45 | -22.85 | -0.8 | 31.92 | 0.15 | within spread |
| 1024x512-c8 | ITL median | 45.67 | 47.26 | 44.52 | 44.45 | 46.46 | 44.49 | -1.97 | -4.2 | 1.59 | 0.07 | better |
| 1024x512-c8 | ITL mean | 52.70 | 54.46 | 52.21 | 51.63 | 53.58 | 51.92 | -1.66 | -3.1 | 1.76 | 0.57 | within spread |
| 1024x512-c8 | ITL p99 | 380.08 | 384.08 | 379.50 | 379.73 | 382.08 | 379.61 | -2.46 | -0.6 | 4.00 | 0.22 | within spread |
| 1024x512-c8 | output tok/s | 271.95 | 256.87 | 282.26 | 267.26 | 264.41 | 274.76 | +10.35 | +3.9 | 15.08 | 15.00 | within spread |
| 1024x512-c8 | MTP acceptance length | 2.06 | 2.07 | 2.13 | 2.00 | 2.07 | 2.06 | -0.00 | -0.2 | 0.01 | 0.13 | within spread |
| 1024x512-c8 | prefill aggregate | 2724.91 | 2696.28 | 2731.24 | 2731.10 | 2710.59 | 2731.17 | +20.57 | +0.8 | 28.63 | 0.14 | within spread |
| 16384x1024-c1 | TTFT median | 5656.36 | 5690.34 | 5643.19 | 5645.30 | 5673.35 | 5644.25 | -29.10 | -0.5 | 33.98 | 2.11 | within spread |
| 16384x1024-c1 | TTFT p99 | 5684.98 | 5721.49 | 5677.84 | 5674.16 | 5703.24 | 5676.00 | -27.24 | -0.5 | 36.52 | 3.68 | within spread |
| 16384x1024-c1 | ITL median | 24.43 | 25.32 | 23.22 | 23.07 | 24.88 | 23.15 | -1.73 | -6.9 | 0.88 | 0.15 | better |
| 16384x1024-c1 | ITL mean | 24.38 | 25.26 | 23.18 | 23.06 | 24.82 | 23.12 | -1.70 | -6.9 | 0.89 | 0.11 | better |
| 16384x1024-c1 | ITL p99 | 24.98 | 25.86 | 23.71 | 23.69 | 25.42 | 23.70 | -1.72 | -6.8 | 0.88 | 0.02 | better |
| 16384x1024-c1 | output tok/s | 64.74 | 63.86 | 67.23 | 60.26 | 64.30 | 63.75 | -0.56 | -0.9 | 0.87 | 6.97 | within spread |
| 16384x1024-c1 | MTP acceptance length | 2.46 | 2.50 | 2.47 | 2.05 | 2.48 | 2.26 | -0.22 | -8.9 | 0.04 | 0.42 | within spread |
| 16384x1024-c1 | prefill aggregate | 2881.98 | 2863.59 | 2885.60 | 2887.47 | 2872.78 | 2886.54 | +13.76 | +0.5 | 18.39 | 1.87 | within spread |
| 16384x1024-c8 | TTFT median | 15726.19 | 8475.83 | 8630.69 | 7445.57 | 12101.01 | 8038.13 | -4062.88 | -33.6 | 7250.36 | 1185.12 | within spread |
| 16384x1024-c8 | TTFT p99 | 50261.24 | 50622.56 | 49803.07 | 49349.35 | 50441.90 | 49576.21 | -865.69 | -1.7 | 361.32 | 453.71 | better |
| 16384x1024-c8 | ITL median | 48.07 | 49.60 | 46.68 | 46.37 | 48.84 | 46.53 | -2.31 | -4.7 | 1.53 | 0.31 | better |
| 16384x1024-c8 | ITL mean | 119.35 | 129.28 | 122.49 | 121.17 | 124.32 | 121.83 | -2.49 | -2.0 | 9.93 | 1.33 | within spread |
| 16384x1024-c8 | ITL p99 | 427.05 | 428.99 | 424.03 | 408.43 | 428.02 | 416.23 | -11.79 | -2.8 | 1.94 | 15.60 | within spread |
| 16384x1024-c8 | output tok/s | 114.38 | 114.01 | 114.98 | 114.25 | 114.19 | 114.62 | +0.42 | +0.4 | 0.37 | 0.74 | within spread |
| 16384x1024-c8 | MTP acceptance length | 2.39 | 2.41 | 2.32 | 2.22 | 2.40 | 2.27 | -0.13 | -5.4 | 0.03 | 0.10 | worse |
| 16384x1024-c8 | prefill aggregate | 2701.01 | 2681.73 | 2725.86 | 2750.92 | 2691.37 | 2738.39 | +47.02 | +1.7 | 19.28 | 25.06 | better |

## Verdict gate

Ship HWQ=4 only if decode ITL median improves by more than the within-condition spread (max of |A1-A2|, |B1-B2|) at c1 AND c8, TTFT/prefill not worse beyond spread, probes/PPL/concurrency PASS.

| cell | ITL median improves beyond spread | TTFT median not worse beyond spread | TTFT p99 not worse | prefill agg not worse beyond spread |
|---|---|---|---|---|
| 1024x512-c1 | YES (-1.61 ms vs spread 1.03) | ok | ok | ok |
| 1024x512-c6 | YES (-1.97 ms vs spread 1.59) | ok | ok | ok |
| 1024x512-c8 | YES (-1.97 ms vs spread 1.59) | ok | ok | ok |
| 16384x1024-c1 | YES (-1.73 ms vs spread 0.88) | ok | ok | ok |
| 16384x1024-c8 | YES (-2.31 ms vs spread 1.53) | ok | ok | ok |

## Probes, PPL, concurrency, dmesg, queue evidence per arm

| arm | HWQ | PROBE | PPL probe | PPL | CONCURRENCY | PROBE-AFTER-LOAD | KFD since READY | Runlist oversubscribed (new dmesg lines) | amdgpu fault lines |
|---|---|---|---|---|---|---|---|---|---|
| A1 | 2 | PASS | PASS | 4.81 | PASS | PASS | Worker_TP0_EP0=+0 Worker_TP1_EP1=+0 Worker_TP2_EP2=+0 Worker_TP3_EP3=+0 -> ok | 0 | 0 |
| B1 | 4 | PASS | PASS | 4.83 | PASS | PASS | Worker_TP0_EP0=+0 Worker_TP1_EP1=+0 Worker_TP2_EP2=+0 Worker_TP3_EP3=+0 -> ok | 0 | 0 |
| B2 | 4 | PASS | PASS | 4.82 | PASS | PASS | Worker_TP0_EP0=+0 Worker_TP1_EP1=+0 Worker_TP2_EP2=+0 Worker_TP3_EP3=+0 -> ok | 0 | 0 |
| A2 | 2 | PASS | PASS | 4.86 | PASS | PASS | Worker_TP0_EP0=+0 Worker_TP1_EP1=+0 Worker_TP2_EP2=+0 Worker_TP3_EP3=+0 -> ok | 0 | 0 |

**A1 (want 2) KFD queues / environ:**

```text
pre-cells: root 705468 environ=2; Worker_TP0_EP0 pid 706586 compute=3 sdma=0 environ=absent; Worker_TP1_EP1 pid 706588 compute=3 sdma=0 environ=absent; Worker_TP2_EP2 pid 706587 compute=3 sdma=0 environ=absent; Worker_TP3_EP3 pid 706589 compute=3 sdma=0 environ=absent
post-cells: root 705468 environ=2; Worker_TP0_EP0 pid 706586 compute=3 sdma=0 environ=absent; Worker_TP1_EP1 pid 706588 compute=3 sdma=0 environ=absent; Worker_TP2_EP2 pid 706587 compute=3 sdma=0 environ=absent; Worker_TP3_EP3 pid 706589 compute=3 sdma=0 environ=absent
```

plan: `export GPU_MAX_HW_QUEUES='2'`

**B1 (want 4) KFD queues / environ:**

```text
pre-cells: root 723481 environ=4; Worker_TP0_EP0 pid 724809 compute=5 sdma=0 environ=absent; Worker_TP1_EP1 pid 724812 compute=5 sdma=0 environ=absent; Worker_TP2_EP2 pid 724810 compute=5 sdma=0 environ=absent; Worker_TP3_EP3 pid 724811 compute=5 sdma=0 environ=absent
post-cells: root 723481 environ=4; Worker_TP0_EP0 pid 724809 compute=5 sdma=0 environ=absent; Worker_TP1_EP1 pid 724812 compute=5 sdma=0 environ=absent; Worker_TP2_EP2 pid 724810 compute=5 sdma=0 environ=absent; Worker_TP3_EP3 pid 724811 compute=5 sdma=0 environ=absent
```

plan: `export GPU_MAX_HW_QUEUES='4'`

**B2 (want 4) KFD queues / environ:**

```text
pre-cells: root 734319 environ=4; Worker_TP0_EP0 pid 735261 compute=5 sdma=0 environ=absent; Worker_TP1_EP1 pid 735263 compute=5 sdma=0 environ=absent; Worker_TP2_EP2 pid 735260 compute=5 sdma=0 environ=absent; Worker_TP3_EP3 pid 735262 compute=5 sdma=0 environ=absent
post-cells: root 734319 environ=4; Worker_TP0_EP0 pid 735261 compute=5 sdma=0 environ=absent; Worker_TP1_EP1 pid 735263 compute=5 sdma=0 environ=absent; Worker_TP2_EP2 pid 735260 compute=5 sdma=0 environ=absent; Worker_TP3_EP3 pid 735262 compute=5 sdma=0 environ=absent
```

plan: `export GPU_MAX_HW_QUEUES='4'`

**A2 (want 2) KFD queues / environ:**

```text
pre-cells: root 753933 environ=2; Worker_TP0_EP0 pid 754957 compute=3 sdma=0 environ=absent; Worker_TP1_EP1 pid 754958 compute=3 sdma=0 environ=absent; Worker_TP2_EP2 pid 754960 compute=3 sdma=0 environ=absent; Worker_TP3_EP3 pid 754959 compute=3 sdma=0 environ=absent
post-cells: root 753933 environ=2; Worker_TP0_EP0 pid 754957 compute=3 sdma=0 environ=absent; Worker_TP1_EP1 pid 754958 compute=3 sdma=0 environ=absent; Worker_TP2_EP2 pid 754960 compute=3 sdma=0 environ=absent; Worker_TP3_EP3 pid 754959 compute=3 sdma=0 environ=absent
```

plan: `export GPU_MAX_HW_QUEUES='2'`

## Run notes

- Tree `~/vllm-hwq-c6b1ff354e` (git archive of opengfx1030/vllm-rdna `rdna_extras` c6b1ff354efdb2d36a004525e0c3dba5af51ebc3, `.sha` file), `.so` set from `~/vllm-audit-build-moe` via `attach_so.sh` (`_rocm_C` md5 3a77f9129ffd; the whole set matches `topo_smoke_2026-11-05/so.md5`); `git diff 47c426fb43 c6b1ff354e -- csrc cmake CMakeLists.txt setup.py rust` is empty. Server venv `venv-7.14.0`, model Qwen3.8-Flash-Next-AWQ-W4A16 snapshot 0939125b, recipe `flashnext-mtp2` TP4, slot B (HIP 6,7,8,9 = V620 0a/0f/14/17, bus-verified before every boot, twice).
- The only difference between arms is the override `GPU_MAX_HW_QUEUES=2|4` on the serve_validate.sh command line (`PRINT=1` plan of every arm exports the wanted value; KFD shows 3 compute queues per worker at 2 and 5 at 4, before and after the cells; `/proc/<pid>/environ` is overwritten by setproctitle and unreadable).
- Cold VLLM_CACHE_ROOT per arm. Cells run from `hook.sh` (LOAD_HOOK) after probes/PPL/concurrency, each inside a KFD evicted_ms window: 20/20 cells `+0` on all four workers, no rerun needed, whole-run evicted_ms since READY +0 in every arm. Quiet windows per the lease README (`quiet/HwqAbba`), boot.lock re-checked before every cell. Seeds per cell identical in all arms (12301..12305, `--temperature 0`, `--ignore-eos`, random dataset).
- Bench client runs from `venv-7.14.0_0.32.0`: `vllm bench serve` aborts at import under `venv-7.14.0` (LLVM "Option 'spirv-expand-step' registered more than once"). It only sends HTTP; same client in all arms. A first A1 boot (aborted-first-attempt/) failed on exactly this at the first cell; it and a half-booted B1 were stopped and the ABBA restarted from A1.
- Peak junction 79-81 C (1 Hz log, `arms/*/temps1s.log`); abort rule never fired.
- ITL median = decode step time as the client sees it; output tok/s and MTP acceptance length move with the sampled tokens (A1 c1 1k acceptance 1.50 vs A2 1.94 vs B 2.07/2.12 explains the +21.7 % tok/s at c1 1k, not the step rate).
- 16k c8 TTFT median is a queue artifact (8 requests in the first wave, 24 queued; A1 15.7 s vs 7.4-8.6 s in the other three arms) and not a decision metric.

## Verdict

**Ship HWQ=4 (gate passed).**

- Decode ITL median, B vs A (mean of A1/A2 vs mean of B1/B2): c1 1k -6.6 % (-1.61 ms, spread A 1.03 / B 0.06), c6 1k -4.5 %, c8 1k -4.2 % (-1.97 ms, spread 1.59 / 0.07), c1 16k -6.9 % (-1.73 ms, 0.88 / 0.15), c8 16k -4.7 % (-2.31 ms, 1.53 / 0.31). Better than the within-condition spread in all five cells; additionally every single B arm beats every single A arm in every cell (ordering A1 < A2 drift in A does not explain it: B1/B2 sit below both).
- TTFT median/p99 and prefill aggregate: all within spread (Δ TTFT median -0.1..-1.4 % at 1k, -0.5 % 16k c1; prefill agg +0.3..+0.8 % at 1k, +0.5 % 16k c1, +1.7 % 16k c8). Not worse anywhere.
- Probes, PPL (4.81/4.83/4.82/4.86), concurrency probe, probe-after-load: PASS in all four arms. Mean ITL gains are smaller (-2.0..-3.6 %) and only c1/16k c1 exceed the spread because mean ITL carries the prefill-interference tail at c6/c8 (ITL p99 within spread).
- MTP acceptance at 16k c8: B 2.27 vs A 2.40 (beyond spread, 'worse'), 16k c1 2.26 vs 2.48 (within spread), 1k cells within spread or higher in B. Output tok/s did not move at 16k c8 (+0.4 %), so this is sampling noise on random prompts rather than a throughput effect; flagged, not a gate metric.
- dmesg `Runlist is getting oversubscribed`: 0 new lines in each of A1, B1, B2, A2 (also 0 new amdgpu fault/timeout/reset/AER lines). The earlier unpaired MonoW4A8 measurement (-7.1/-5.5/-3.9 % at c1/c6/c8) is reproduced (-6.6/-4.5/-4.2 %); the 11-02 ABBA's contrary result is not.
