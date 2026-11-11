# Mono + W4A8 comparison

- tree `/home/chenco_adm/w4a8_runs/mono-w4a8/trees/tree-5b8d1e5570` at `5b8d1e55707476d8191b3c9308d25a54ef2ecaf4`
- `.so` provenance: extensions built 2026-11-05 09:01 CET from tree build-179ebec43b (git archive of 179ebec43b8e6a413eaf2b09876af3a64e844338) in the shared incremental dir ~/vllm-audit-build-moe under ~/w4a8_runs/audit/build.lock (MAX_JOBS=8, nice 19, CPU-only); log ~/w4a8_runs/mono-w4a8/build/build_179ebec43b.log.
csrc/, cmake/, CMakeLists.txt, setup.py of this SHA are byte-identical to origin/rdna_extras 47c426fb43 and to 179ebec43b (git diff --stat empty), so these .so files are the 47c426fb43 build.
- arms: c: --monokernel + W4A8 (4 HW queues) (try 1); a: production: W4A8 prefill, no --monokernel (2 HW queues) (try 1); c2: --monokernel + W4A8 (4 HW queues) (try 1); a2: production: W4A8 prefill, no --monokernel (2 HW queues) (try 1); d: --monokernel + W4A8, hw_queues 0 (2 HW queues) (try 1); e: production at GPU_MAX_HW_QUEUES=4 (control) (try 1); b: --monokernel, W4A8 off (mono raises to 4 HW queues) (try 1)

## Arms

| arm | boot s | model GiB | KV tokens | mono lines | W4A8 part | sdot4 active | K2 full lines | queue raise | eviction (whole hook) | invalid cells | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| c | 290 | 20.56 | 266,945 | 4 | 4 | 4 | 8 | 1 | ok | none | DONE |
| a | 281 | 20.56 | 266,945 | 0 | 0 | 4 | 0 | 0 | ok | none | DONE |
| c2 | 191 | 20.56 | 266,945 | 4 | 4 | 4 | 8 | 1 | ok | none | DONE |
| a2 | 190 | 20.56 | 266,945 | 0 | 0 | 4 | 0 | 0 | ok | none | DONE |
| d | 240 | 20.56 | 266,945 | 4 | 4 | 4 | 8 | 0 | ok | none | DONE |
| e | 190 | 20.56 | 266,945 | 0 | 0 | 4 | 0 | 0 | ok | none | DONE |
| b | 190 | 20.56 | 266,945 | 4 | 0 | 0 | 8 | 1 | ok | none | DONE |

## Host during the arm (load, memory lock, late peers, temperatures)

| arm | load avg at start / end of the hook | top CPU processes at start / end | new AER lines | pinned PLE table | VmLck GiB, lowest worker (start / end) | RLIMIT_MEMLOCK | rdna_ar late peers | driver evictions in rank logs | KFD compute queues per worker (max over samples) | junction C mean/max per GPU |
|---|---|---|---:|---|---|---|---:|---:|---|---|
| c | 3.78 7.54 13.55 / 13.57 11.74 11.45 | 310% python, 99.3% clang-23 / 185% VLLM::Worker_TP | 0 | 23.84 GiB, 4/4 mlocked | 23.8 / 23.8 | unlimited | 0 | 0 | n/a | hip2:55/75 hip3:55/76 hip4:58/84 hip5:61/87 |
| a | 10.99 11.27 11.30 / 8.79 9.13 9.49 | 99.7% tensorfold-nati, 7.0% buildkitd / 300% python3, 185% VLLM::Worker_TP | 0 | 23.84 GiB, 4/4 mlocked | 23.8 / 23.8 | unlimited | 0 | 0 | 3 (n=47) | hip2:57/75 hip3:56/75 hip4:62/84 hip5:65/87 |
| c2 | 7.20 8.71 9.34 / 8.45 9.31 9.60 | 100% tensorfold-nati, 10.3% buildkitd / 188% VLLM::Worker_TP | 0 | 23.84 GiB, 4/4 mlocked | 23.8 / 23.8 | unlimited | 0 | 0 | 5 (n=45) | hip2:58/75 hip3:57/76 hip4:64/84 hip5:67/87 |
| a2 | 22.31 12.17 10.52 / 9.05 9.30 9.55 | 250% python3, 96.9% tensorfold-nati / 191% VLLM::Worker_TP | 0 | 23.84 GiB, 4/4 mlocked | 23.8 / 23.8 | unlimited | 0 | 0 | 3 (n=45) | hip2:58/75 hip3:57/75 hip4:63/83 hip5:66/86 |
| d | 7.67 8.94 9.42 / 11.40 14.10 11.56 | 163% python3, 94.0% tensorfold-nati / 186% VLLM::Worker_TP, 137% VLLM::Worker_TP | 0 | 23.84 GiB, 4/4 mlocked | 23.8 / 23.8 | unlimited | 0 | 0 | 3 (n=48) | hip2:59/76 hip3:58/76 hip4:64/83 hip5:68/87 |
| e | 10.14 13.54 11.45 / 7.78 9.02 9.60 | 96.4% tensorfold-nati / 189% VLLM::Worker_TP, 137% VLLM::Worker_TP | 0 | 23.84 GiB, 4/4 mlocked | 23.8 / 23.8 | unlimited | 0 | 0 | 5 (n=45) | hip2:59/76 hip3:58/76 hip4:65/84 hip5:69/87 |
| b | 6.51 8.60 9.44 / 9.73 10.71 10.15 | 3051% python, 3.5% sshd-session / 1822% python, 197% VLLM::Worker_TP | 0 | 23.84 GiB, 4/4 mlocked | 23.8 / 23.8 | unlimited | 0 | 0 | 5 (n=48) | hip2:60/77 hip3:59/78 hip4:66/85 hip5:70/89 |

Junction temperatures come from `temps.csv` (`gpu_temps.py`, every 15 s over the lease); they are logged only, nothing gates on them. A late peer or a driver eviction in a rank log inside a measured window is a host stall, not the code under test. KFD compute queues per worker come from `queues.jsonl` (`queue_probe.py`, every 20 s): the budget (`GPU_MAX_HW_QUEUES`) cannot be read from a worker (setproctitle overwrites its environ), so the queues CLR actually created are the evidence; the engine's raise line is in the arm table.

## Performance (valid cells only)

| metric | c | a | c2 | a2 | d | e | b |
|---|---:|---:|---:|---:|---:|---:|---:|
| TTFT 4k solo, median (s) | 1.42 | 1.44 | 1.42 | 1.43 | 1.44 | 1.43 | 1.83 |
| prefill 4k (tok/s/req) | 2892 | 2857 | 2884 | 2878 | 2855 | 2875 | 2242 |
| TTFT 16k solo, median (s) | 5.61 | 5.67 | 5.62 | 5.65 | 5.68 | 5.66 | 7.31 |
| prefill 16k (tok/s/req) | 2925 | 2896 | 2919 | 2906 | 2887 | 2902 | 2243 |
| 16k c8 aggregate prefill (tok/s) | 2661 | 2642 | 2664 | 2635 | 2655 | 2660 | 2117 |
| 16k c8 mean TTFT (s) | 17.54 | 18.80 | 18.73 | 19.38 | 19.45 | 18.98 | 23.59 |
| 16k/1k c8 decode (tok/s/req) | 20.2 | 21.0 | 21.1 | 20.5 | 21.2 | 21.1 | 18.0 |
| c1 ITL median (ms) | 23.7 | 24.2 | 23.7 | 24.2 | 24.7 | 22.5 | 23.7 |
| c1 decode (tok/s/req) | 112.4 | 90.7 | 107.5 | 98.1 | 102.3 | 112.8 | 108.5 |
| c6 ITL median (ms) | 42.1 | 43.7 | 42.2 | 43.4 | 44.0 | 41.2 | 43.0 |
| c6 decode (tok/s/req) | 40.9 | 43.7 | 43.9 | 42.1 | 42.8 | 45.1 | 42.0 |
| c8 ITL median (ms) | 45.5 | 45.9 | 45.9 | 46.1 | 46.8 | 44.2 | 45.7 |
| c8 decode (tok/s/req) | 36.9 | 37.5 | 39.4 | 37.7 | 37.4 | 39.9 | 38.8 |
| c8 output (tok/s total) | 259 | 267 | 276 | 260 | 263 | 281 | 270 |
| c1 accepted length | 2.66 | 2.20 | 2.55 | 2.37 | 2.53 | 2.53 | 2.57 |
| c8 accepted length | 1.95 | 2.01 | 2.12 | 2.03 | 2.04 | 2.08 | 2.18 |

| metric | production (a, a2) | production, 4 queues (e) | mono, W4A16 (b) | mono + W4A8 (c, c2) | mono + W4A8, 2 queues (d) | mono+W4A8 vs prod | mono vs prod | mono+W4A8 2q vs prod | prod 4q vs 2q | mono+W4A8 vs prod 4q | W4A8 inside mono | 4q vs 2q (W4A8) | noise prod % | noise mono+W4A8 % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TTFT 4k solo, median (s) | 1.43 | 1.43 | 1.83 | 1.42 | 1.44 | -0.6% ~ | +27.9% - | +0.5% ~ | -0.3% ~ | -0.3% ~ | -22.3% + | -1.1% + | 0.8 | 0.4 |
| prefill 4k (tok/s/req) | 2867 | 2875 | 2242 | 2888 | 2855 | +0.7% ~ | -21.8% - | -0.4% ~ | +0.3% ~ | +0.4% ~ | +28.8% + | +1.1% + | 0.7 | 0.3 |
| TTFT 16k solo, median (s) | 5.66 | 5.66 | 7.31 | 5.62 | 5.68 | -0.7% ~ | +29.1% - | +0.5% ~ | +0.0% ~ | -0.8% ~ | -23.1% + | -1.2% + | 0.3 | 0.2 |
| prefill 16k (tok/s/req) | 2901 | 2902 | 2243 | 2922 | 2887 | +0.7% ~ | -22.7% - | -0.5% ~ | +0.0% ~ | +0.7% ~ | +30.3% + | +1.2% + | 0.3 | 0.2 |
| 16k c8 aggregate prefill (tok/s) | 2638 | 2660 | 2117 | 2663 | 2655 | +0.9% ~ | -19.8% - | +0.6% ~ | +0.8% ~ | +0.1% ~ | +25.8% + | +0.3% ~ | 0.3 | 0.1 |
| 16k c8 mean TTFT (s) | 19.09 | 18.98 | 23.59 | 18.14 | 19.45 | -5.0% ~ | +23.6% - | +1.9% ~ | -0.6% ~ | -4.5% ~ | -23.1% + | -6.8% + | 3.1 | 6.6 |
| 16k/1k c8 decode (tok/s/req) | 20.8 | 21.1 | 18.0 | 20.6 | 21.2 | -0.5% ~ | -13.4% - | +2.2% ~ | +1.4% ~ | -1.9% ~ | +14.9% + | -2.7% ~ | 2.2 | 4.7 |
| c1 ITL median (ms) | 24.2 | 22.5 | 23.7 | 23.7 | 24.7 | -1.9% + | -1.9% + | +2.1% - | -7.1% + | +5.6% - | -0.1% ~ | -3.9% + | 0.2 | 0.1 |
| c1 decode (tok/s/req) | 94.4 | 112.8 | 108.5 | 110.0 | 102.3 | +16.5% + | +15.0% + | +8.3% + | +19.5% + | -2.5% ~ | +1.3% ~ | +7.5% ~ | 7.8 | 4.4 |
| c6 ITL median (ms) | 43.6 | 41.2 | 43.0 | 42.2 | 44.0 | -3.2% + | -1.3% + | +1.0% ~ | -5.5% + | +2.4% - | -2.0% + | -4.1% + | 0.5 | 0.3 |
| c6 decode (tok/s/req) | 42.9 | 45.1 | 42.0 | 42.4 | 42.8 | -1.1% ~ | -2.1% ~ | -0.1% ~ | +5.2% ~ | -5.9% ~ | +1.1% ~ | -1.0% ~ | 3.7 | 7.0 |
| c8 ITL median (ms) | 46.0 | 44.2 | 45.7 | 45.7 | 46.8 | -0.6% ~ | -0.6% ~ | +1.8% - | -3.9% + | +3.4% - | -0.0% ~ | -2.3% + | 0.6 | 1.0 |
| c8 decode (tok/s/req) | 37.6 | 39.9 | 38.8 | 38.2 | 37.4 | +1.6% ~ | +3.1% ~ | -0.5% ~ | +6.1% ~ | -4.3% ~ | -1.5% ~ | +2.1% ~ | 0.4 | 6.6 |
| c8 output (tok/s total) | 264 | 281 | 270 | 268 | 263 | +1.5% ~ | +2.6% ~ | -0.0% ~ | +6.6% ~ | -4.7% ~ | -1.0% ~ | +1.5% ~ | 3.0 | 6.6 |
| c1 accepted length | 2.28 | 2.53 | 2.57 | 2.61 | 2.53 | +14.2% + | +12.7% + | +10.6% + | +10.9% + | +3.0% ~ | +1.3% ~ | +3.2% ~ | 7.6 | 4.2 |
| c8 accepted length | 2.02 | 2.08 | 2.18 | 2.04 | 2.04 | +0.9% ~ | +8.0% ~ | +1.2% ~ | +3.2% ~ | -2.2% ~ | -6.5% ~ | -0.3% ~ | 0.8 | 8.3 |

`+` better than the reference beyond the noise, `-` worse beyond it, `~` within the noise (the larger of 1 % and the boot-to-boot spread of the repeated arms, a/a2 and c/c2). Each cell is one boot per arm; a spread needs the repeats.

## Probes

| arm | greedy probe | PPL probe (261 tok) | prefix | concurrency | probe after load | prefix cold/warm TTFT (s) | tracebacks |
|---|---|---|---|---|---|---|---:|
| c | PASS | PASS (4.84) | PASS | PASS | PASS | 4.42 / 0.28 | 0 |
| a | PASS | PASS (4.83) | PASS | PASS | PASS | 4.36 / 0.29 | 0 |
| c2 | PASS | PASS (4.83) | PASS | PASS | PASS | 4.37 / 0.29 | 0 |
| a2 | PASS | PASS (4.82) | PASS | PASS | PASS | 4.36 / 0.28 | 0 |
| d | PASS | PASS (4.74) | PASS | PASS | PASS | 4.38 / 0.29 | 0 |
| e | PASS | PASS (4.77) | PASS | PASS | PASS | 4.37 / 0.29 | 0 |
| b | PASS | PASS (4.81) | PASS | PASS | PASS | 5.64 / 0.29 | 0 |

## Teacher-forced windows (prefill path, W4A8 MoE when on)

| pair (ref, cand) | domain | windows | PPL ref | PPL cand | dNLL [95% CI] | KL nats/token [95% CI] | top-1 | identical tokens |
|---|---|---:|---:|---:|---|---|---:|---:|
| a2 vs a (boot floor, W4A8) | docs | 32 | 3.3941 | 3.3949 | +0.00022 [-0.00221, +0.00259] | 0.01393 [0.01159, 0.01639] | 0.9598 | 0.5% |
| a2 vs a (boot floor, W4A8) | mix | 32 | 2.8753 | 2.8614 | -0.00486 [-0.02282, +0.00867] | 0.02093 [0.01402, 0.03081] | 0.9658 | 0.1% |
| c2 vs c (boot floor, mono+W4A8) | docs | 32 | 3.3948 | 3.3899 | -0.00145 [-0.00324, +0.00033] | 0.01346 [0.01132, 0.01578] | 0.9590 | 0.5% |
| c2 vs c (boot floor, mono+W4A8) | mix | 32 | 2.8558 | 2.8597 | +0.00137 [-0.00668, +0.00926] | 0.01859 [0.01360, 0.02384] | 0.9667 | 0.2% |
| c vs a (headline) | docs | 32 | 3.3941 | 3.3948 | +0.00019 [-0.00162, +0.00201] | 0.01346 [0.01118, 0.01590] | 0.9612 | 0.5% |
| c vs a (headline) | mix | 32 | 2.8753 | 2.8558 | -0.00682 [-0.02808, +0.01241] | 0.02205 [0.01502, 0.03037] | 0.9644 | 0.1% |
| c2 vs a2 | docs | 32 | 3.3949 | 3.3899 | -0.00148 [-0.00346, +0.00041] | 0.01379 [0.01147, 0.01630] | 0.9603 | 0.5% |
| c2 vs a2 | mix | 32 | 2.8614 | 2.8597 | -0.00059 [-0.00703, +0.00580] | 0.01737 [0.01299, 0.02208] | 0.9677 | 0.1% |
| b vs a | docs | 32 | 3.3941 | 3.3901 | -0.00119 [-0.00316, +0.00071] | 0.01355 [0.01127, 0.01593] | 0.9596 | 0.5% |
| b vs a | mix | 32 | 2.8753 | 2.8901 | +0.00515 [-0.00877, +0.01830] | 0.02121 [0.01545, 0.02763] | 0.9651 | 0.1% |
| c vs b (W4A8 inside mono) | docs | 32 | 3.3901 | 3.3948 | +0.00138 [-0.00030, +0.00312] | 0.01354 [0.01117, 0.01610] | 0.9600 | 0.6% |
| c vs b (W4A8 inside mono) | mix | 32 | 2.8901 | 2.8558 | -0.01197 [-0.02765, +0.00202] | 0.02260 [0.01522, 0.03066] | 0.9646 | 0.1% |
| d vs a | docs | 32 | 3.3941 | 3.3944 | +0.00007 [-0.00190, +0.00197] | 0.01350 [0.01119, 0.01583] | 0.9604 | 0.5% |
| d vs a | mix | 32 | 2.8753 | 2.8532 | -0.00773 [-0.03094, +0.00910] | 0.02423 [0.01521, 0.03754] | 0.9650 | 0.1% |
| e vs a | docs | 32 | 3.3941 | 3.3946 | +0.00013 [-0.00153, +0.00175] | 0.01363 [0.01132, 0.01613] | 0.9604 | 0.6% |
| e vs a | mix | 32 | 2.8753 | 2.8686 | -0.00232 [-0.01944, +0.01121] | 0.02352 [0.01525, 0.03562] | 0.9648 | 0.2% |
| c repeat (same boot) | docs | 16 | 4.1936 | 4.1946 | +0.00023 [-0.00233, +0.00275] | 0.01579 [0.01229, 0.01952] | 0.9523 | 0.4% |
| c repeat (same boot) | mix | 16 | 2.7584 | 2.7747 | +0.00588 [-0.00369, +0.01646] | 0.01327 [0.00879, 0.01911] | 0.9718 | 0.0% |
| a repeat (same boot) | docs | 16 | 4.1929 | 4.1957 | +0.00066 [-0.00205, +0.00337] | 0.01548 [0.01222, 0.01870] | 0.9523 | 0.4% |
| a repeat (same boot) | mix | 16 | 2.7729 | 2.7575 | -0.00556 [-0.01195, +0.00121] | 0.01277 [0.00933, 0.01682] | 0.9705 | 0.0% |
| c2 repeat (same boot) | docs | 16 | 4.1946 | 4.1952 | +0.00016 [-0.00291, +0.00317] | 0.01569 [0.01238, 0.01919] | 0.9533 | 0.4% |
| c2 repeat (same boot) | mix | 16 | 2.7535 | 2.7770 | +0.00848 [+0.00308, +0.01443] | 0.01104 [0.00859, 0.01354] | 0.9727 | 0.0% |
| a2 repeat (same boot) | docs | 16 | 4.1967 | 4.1907 | -0.00142 [-0.00440, +0.00190] | 0.01580 [0.01237, 0.01936] | 0.9522 | 0.3% |
| a2 repeat (same boot) | mix | 16 | 2.7671 | 2.7665 | -0.00022 [-0.01252, +0.01077] | 0.01317 [0.00869, 0.01829] | 0.9710 | 0.0% |
| d repeat (same boot) | docs | 16 | 4.1948 | 4.1980 | +0.00077 [-0.00200, +0.00408] | 0.01570 [0.01237, 0.01913] | 0.9519 | 0.4% |
| d repeat (same boot) | mix | 16 | 2.7700 | 2.7684 | -0.00058 [-0.00879, +0.00572] | 0.01179 [0.00776, 0.01788] | 0.9728 | 0.0% |
| e repeat (same boot) | docs | 16 | 4.1930 | 4.1904 | -0.00063 [-0.00330, +0.00161] | 0.01591 [0.01251, 0.01941] | 0.9512 | 0.4% |
| e repeat (same boot) | mix | 16 | 2.7680 | 2.7640 | -0.00145 [-0.01005, +0.00710] | 0.01164 [0.00867, 0.01491] | 0.9716 | 0.0% |
| b repeat (same boot) | docs | 16 | 4.1873 | 4.1915 | +0.00100 [-0.00151, +0.00346] | 0.01040 [0.00798, 0.01285] | 0.9622 | 0.5% |
| b repeat (same boot) | mix | 16 | 2.7701 | 2.7748 | +0.00169 [-0.00385, +0.00687] | 0.00828 [0.00590, 0.01116] | 0.9777 | 0.0% |

Mechanical reading of `c vs a` against the boot-to-boot floors (KL margin 0.003):

- docs: KL(c vs a) 0.01346 vs floor 0.01393 + 0.003 -> within noise; dNLL +0.00019 CI contains 0 -> PASS
- mix: KL(c vs a) 0.02205 vs floor 0.02093 + 0.003 -> within noise; dNLL -0.00682 CI contains 0 -> PASS

## Decode logprob parity (greedy, top-5, solo and c8)

| pair | identical-prefix tokens | worst abs dlogprob on the shared prefix |
|---|---:|---:|
| a vs a2 | 293/725 | 1.2043 |
| c vs c2 | 617/788 | 0.6304 |
| a vs c | 559/741 | 0.5470 |
| a vs b | 513/678 | 0.5137 |
| b vs c | 483/725 | 0.6492 |
| a vs d | 491/725 | 0.2918 |
